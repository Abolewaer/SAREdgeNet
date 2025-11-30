import sys
import os
import cv2
import numpy as np
from scipy import ndimage
import time
import multiprocessing as mp
from pathlib import Path
import queue
import threading
import rasterio
from rasterio.enums import ColorInterp

from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QLabel, QLineEdit, QSpinBox,
                             QProgressBar, QTextEdit, QFileDialog, QGroupBox,
                             QCheckBox, QComboBox, QGridLayout, QMessageBox,
                             QSplitter, QFrame, QScrollArea, QTabWidget)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap, QPalette, QColor


class GeoTIFFHandler:
    """Core GeoTIFF file processing class, uses rasterio to preserve geographic information"""

    @staticmethod
    def read_geotiff(image_path):
        """Read GeoTIFF file, return image data and geographic metadata"""
        try:
            with rasterio.open(image_path) as src:
                # Read all band data
                image_data = src.read()  # Shape: (bands, height, width)

                # Get geographic metadata
                profile = src.profile.copy()

                # Convert to OpenCV format (height, width, bands)
                if len(image_data.shape) == 3:
                    image = np.transpose(image_data, (1, 2, 0))
                else:
                    image = image_data

                # Ensure 3-channel BGR format (OpenCV format)
                if image.shape[-1] == 3:
                    # rasterio reads RGB, convert to BGR
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                elif len(image.shape) == 2:
                    # Single channel to 3 channels
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

                return image, profile

        except Exception as e:
            print(f"Error reading GeoTIFF file {image_path}: {str(e)}")
            return None, None

    @staticmethod
    def write_geotiff(image_path, image_data, profile):
        """Write GeoTIFF file, preserve geographic information"""
        try:
            # Process image data format
            if len(image_data.shape) == 3:
                # BGR to RGB (rasterio uses RGB)
                if image_data.shape[-1] == 3:
                    image_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
                    # Convert to rasterio format (bands, height, width)
                    image_for_write = np.transpose(image_rgb, (2, 0, 1))
                elif image_data.shape[-1] == 4:
                    # BGRA to RGBA
                    image_rgba = cv2.cvtColor(image_data, cv2.COLOR_BGRA2RGBA)
                    image_for_write = np.transpose(image_rgba, (2, 0, 1))
                else:
                    image_for_write = image_data
            else:
                image_for_write = image_data

            # Update profile
            profile.update({
                'height': image_for_write.shape[-2],
                'width': image_for_write.shape[-1],
                'count': image_for_write.shape[0] if len(image_for_write.shape) == 3 else 1,
                'dtype': image_for_write.dtype
            })

            # Write file
            with rasterio.open(image_path, 'w', **profile) as dst:
                if len(image_for_write.shape) == 3:
                    for i in range(image_for_write.shape[0]):
                        dst.write(image_for_write[i], i + 1)
                else:
                    dst.write(image_for_write, 1)

            return True

        except Exception as e:
            print(f"Error writing GeoTIFF file {image_path}: {str(e)}")
            return False


class ContourProcessor:
    """Core contour processing class - uses rasterio to preserve geographic information"""

    @staticmethod
    def vectorized_gap_analysis(labeled_gaps, gray_image, mean_threshold, std_threshold, small_gap_threshold):
        """Batch analyze gaps using vectorized operations"""
        unique_labels = np.unique(labeled_gaps)[1:]  # Exclude background label 0

        if len(unique_labels) == 0:
            return np.array([]), np.array([]), np.array([])

        # Pre-allocate arrays
        gap_sizes = np.zeros(len(unique_labels), dtype=np.int32)
        gap_means = np.zeros(len(unique_labels), dtype=np.float32)
        gap_stds = np.zeros(len(unique_labels), dtype=np.float32)

        # Batch calculate statistics
        for i, label in enumerate(unique_labels):
            mask = labeled_gaps == label
            pixels = gray_image[mask]
            gap_sizes[i] = np.sum(mask)
            gap_means[i] = np.mean(pixels)
            gap_stds[i] = np.std(pixels)

        # Vectorized condition judgment
        small_gaps = gap_sizes <= small_gap_threshold
        condition_gaps = (gap_means <= mean_threshold) & (gap_stds <= std_threshold) & (~small_gaps)
        fill_gaps = small_gaps | condition_gaps

        return unique_labels[fill_gaps], gap_sizes[fill_gaps], small_gaps[fill_gaps]

    @staticmethod
    def batch_process_contours(image, red_mask, contours, mean_threshold, std_threshold,
                               small_gap_threshold, fill_color, batch_size=50, progress_queue=None, process_id=0):
        """Batch process contours, only fill non-red regions inside contours - optimized version"""
        result_image = image.copy()
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        total_filled = 0
        total_gaps_analyzed = 0
        small_gaps_filled = 0
        condition_gaps_filled = 0
        total_contours = len(contours)

        # Filter out too small contours to avoid invalid calculations
        min_contour_area = max(10, small_gap_threshold // 4)  # Dynamic minimum area
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_contour_area]

        if len(filtered_contours) < len(contours):
            if progress_queue:
                try:
                    progress_queue.put({
                        'type': 'contour_filtered',
                        'process_id': process_id,
                        'original': len(contours),
                        'filtered': len(filtered_contours),
                        'tool': 'filler'
                    })
                except:
                    pass

        total_contours = len(filtered_contours)
        contours = filtered_contours

        # Process contours in batches
        contour_index = 0
        for batch_start in range(0, len(contours), batch_size):
            batch_end = min(batch_start + batch_size, len(contours))
            batch_contours = contours[batch_start:batch_end]

            # Process each contour in current batch
            for contour in batch_contours:
                contour_index += 1

                # Report contour processing progress
                if progress_queue and contour_index % 10 == 0:  # Reduce communication frequency
                    try:
                        progress_queue.put({
                            'type': 'contour_progress',
                            'process_id': process_id,
                            'current': contour_index,
                            'total': total_contours,
                            'tool': 'filler'
                        })
                    except:
                        pass  # Ignore queue errors

                # Use bounding box optimization - only process region around contour
                x, y, w, h = cv2.boundingRect(contour)

                # Add margin to ensure all relevant pixels are included
                margin = 5
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(red_mask.shape[1], x + w + margin)
                y2 = min(red_mask.shape[0], y + h + margin)

                # Only create mask within bounding box
                bbox_shape = (y2 - y1, x2 - x1)
                bbox_contour_mask = np.zeros(bbox_shape, dtype=np.uint8)

                # Adjust contour coordinates to bounding box
                adjusted_contour = contour.copy()
                adjusted_contour[:, 0, 0] -= x1  # x coordinate offset
                adjusted_contour[:, 0, 1] -= y1  # y coordinate offset

                cv2.fillPoly(bbox_contour_mask, [adjusted_contour], 255)

                # 在边界框内处理空隙检测
                bbox_red_mask = red_mask[y1:y2, x1:x2]
                bbox_gray = gray_image[y1:y2, x1:x2]

                # 关键修正：只考虑轮廓内部的非红色区域（在边界框内）
                internal_gaps = (bbox_contour_mask > 0) & (bbox_red_mask == 0)

                # 只有当存在内部空隙时才进行处理
                if np.any(internal_gaps):
                    # 对当前轮廓的内部空隙进行连通域标记
                    labeled_gaps, num_gaps = ndimage.label(internal_gaps)

                    if num_gaps > 0:
                        fill_labels, gap_sizes, is_small_gaps = ContourProcessor.vectorized_gap_analysis(
                            labeled_gaps, bbox_gray, mean_threshold, std_threshold, small_gap_threshold
                        )

                        total_gaps_analyzed += num_gaps

                        # 批量填充（在原图坐标系中）
                        for i, label in enumerate(fill_labels):
                            bbox_mask = labeled_gaps == label
                            # 将边界框掩码映射回原图坐标
                            full_mask = np.zeros(red_mask.shape, dtype=bool)
                            full_mask[y1:y2, x1:x2] = bbox_mask

                            result_image[full_mask] = fill_color
                            total_filled += gap_sizes[i]

                            if is_small_gaps[i]:
                                small_gaps_filled += 1
                            else:
                                condition_gaps_filled += 1

        # 发送最终进度
        if progress_queue:
            try:
                progress_queue.put({
                    'type': 'contour_progress',
                    'process_id': process_id,
                    'current': total_contours,
                    'total': total_contours,
                    'tool': 'filler'
                })
            except:
                pass

        return result_image, total_filled, total_gaps_analyzed, small_gaps_filled, condition_gaps_filled

    @staticmethod
    def detect_and_fill_internal_gaps(image_path, output_path, mean_threshold=50, std_threshold=30,
                                      small_gap_threshold=20, use_green_fill=False, batch_size=50,
                                      progress_queue=None, process_id=0):
        """检测并填充轮廓内部空隙的主函数 - 使用rasterio保留地理信息"""
        try:
            start_time = time.time()

            # 使用rasterio读取图像和地理信息
            image, profile = GeoTIFFHandler.read_geotiff(image_path)
            if image is None or profile is None:
                return False, 0, 0, 0, 0

            # 预处理步骤：将所有红色主导的像素标准化为纯红色(255,0,0)
            if progress_queue:
                try:
                    progress_queue.put({
                        'type': 'preprocessing',
                        'process_id': process_id,
                        'message': '正在标准化红色像素...',
                        'tool': 'filler'
                    })
                except:
                    pass

            # 分离BGR通道
            blue_channel = image[:, :, 0]
            green_channel = image[:, :, 1]
            red_channel = image[:, :, 2]

            # 找出红色通道大于其他通道的像素
            red_dominant_mask = (red_channel > green_channel) & (red_channel > blue_channel)

            # 统计标准化前的红色像素
            original_red_pixels = np.sum(red_dominant_mask)

            # 将红色主导的像素设置为纯红色
            image[red_dominant_mask] = [0, 0, 255]  # BGR格式：蓝=0，绿=0，红=255

            if progress_queue:
                try:
                    progress_queue.put({
                        'type': 'preprocessing_complete',
                        'process_id': process_id,
                        'standardized_pixels': int(original_red_pixels),
                        'tool': 'filler'
                    })
                except:
                    pass

            # 提取红色像素掩膜
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # 红色的HSV范围
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)

            red_pixel_count = np.sum(red_mask > 0)
            if red_pixel_count == 0:
                success = GeoTIFFHandler.write_geotiff(output_path, image, profile)
                return success, 0, 0, time.time() - start_time, 0

            # 查找红色轮廓
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_count = len(contours)

            if contour_count == 0:
                success = GeoTIFFHandler.write_geotiff(output_path, image, profile)
                return success, 0, 0, time.time() - start_time, 0

            # 报告找到的轮廓数量
            if progress_queue:
                try:
                    progress_queue.put({
                        'type': 'contour_found',
                        'process_id': process_id,
                        'count': contour_count,
                        'tool': 'filler'
                    })
                except:
                    pass

            # 设置填充颜色
            fill_color = [0, 255, 0] if use_green_fill else [0, 0, 255]  # 绿色或红色 (BGR格式)

            # 批量处理轮廓
            result_image, total_filled_pixels, total_gaps_analyzed, small_gaps_filled, condition_gaps_filled = ContourProcessor.batch_process_contours(
                image, red_mask, contours, mean_threshold, std_threshold,
                small_gap_threshold, fill_color, batch_size, progress_queue, process_id
            )

            # 使用rasterio保存结果，保留地理信息
            success = GeoTIFFHandler.write_geotiff(output_path, result_image, profile)

            processing_time = time.time() - start_time

            return success, total_filled_pixels, total_gaps_analyzed, processing_time, contour_count

        except Exception as e:
            print(f"进程 {process_id} 处理文件 {image_path} 时出错: {str(e)}")
            return False, 0, 0, 0, 0


class RedRegionCleaner:
    """红色连通域清理核心类 - 使用rasterio保留地理信息"""

    @staticmethod
    def detect_red_regions(image):
        """检测所有红色连通域"""
        # 转换为HSV进行红色检测
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 红色的HSV范围
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        return red_mask

    @staticmethod
    def process_contours_with_bbox_optimization(image, red_mask, contours, min_area_threshold,
                                                white_color, progress_queue=None, process_id=0):
        """使用真正连通域的边界框优化处理"""
        result_image = image.copy()

        # 先进行全图连通域分析，获得真正的连通域
        if progress_queue:
            try:
                progress_queue.put({
                    'type': 'analyzing_components',
                    'process_id': process_id,
                    'message': '正在分析真正的连通域...',
                    'tool': 'cleaner'
                })
            except:
                pass

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            red_mask, connectivity=8
        )

        total_components = num_labels - 1  # 减去背景标签

        if progress_queue:
            try:
                progress_queue.put({
                    'type': 'components_found',
                    'process_id': process_id,
                    'count': total_components,
                    'tool': 'cleaner'
                })
            except:
                pass

        if total_components == 0:
            return result_image, {
                'removed_count': 0, 'kept_count': 0,
                'removed_pixels': 0, 'kept_pixels': 0,
                'total_components': 0
            }

        total_removed_count = 0
        total_kept_count = 0
        total_removed_pixels = 0
        total_kept_pixels = 0

        # 对每个真正的连通域进行边界框优化处理
        for component_id in range(1, num_labels):  # 跳过背景标签0
            # 报告处理进度
            if progress_queue and component_id % 10 == 0:
                try:
                    progress_queue.put({
                        'type': 'component_progress',
                        'process_id': process_id,
                        'current': component_id,
                        'total': total_components,
                        'tool': 'cleaner'
                    })
                except:
                    pass

            # 获取当前连通域的面积（这是真正的连通域面积）
            area = stats[component_id, cv2.CC_STAT_AREA]

            # 判断是否需要删除
            if area < min_area_threshold:
                # 获取当前连通域的边界框
                x = stats[component_id, cv2.CC_STAT_LEFT]
                y = stats[component_id, cv2.CC_STAT_TOP]
                w = stats[component_id, cv2.CC_STAT_WIDTH]
                h = stats[component_id, cv2.CC_STAT_HEIGHT]

                # 添加小边距，确保完整包含连通域
                margin = 2
                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(labels.shape[1], x + w + margin)
                y2 = min(labels.shape[0], y + h + margin)

                # 在边界框内找到当前连通域的像素
                bbox_labels = labels[y1:y2, x1:x2]
                component_mask_bbox = (bbox_labels == component_id)

                # 将边界框内的掩码映射回原图坐标
                full_mask = np.zeros(labels.shape, dtype=bool)
                full_mask[y1:y2, x1:x2] = component_mask_bbox

                # 将小连通域替换为指定颜色
                result_image[full_mask] = white_color

                total_removed_count += 1
                total_removed_pixels += area
            else:
                # 保留的大连通域
                total_kept_count += 1
                total_kept_pixels += area

        # 发送最终进度
        if progress_queue:
            try:
                progress_queue.put({
                    'type': 'component_progress',
                    'process_id': process_id,
                    'current': total_components,
                    'total': total_components,
                    'tool': 'cleaner'
                })
            except:
                pass

        stats_result = {
            'removed_count': total_removed_count,
            'kept_count': total_kept_count,
            'removed_pixels': total_removed_pixels,
            'kept_pixels': total_kept_pixels,
            'total_components': total_components
        }

        return result_image, stats_result

    @staticmethod
    def clean_small_red_regions(image_path, output_path, min_area_threshold=100,
                                white_color=(255, 255, 255), progress_queue=None, process_id=0):
        """清理小红色连通域的主函数 - 使用rasterio保留地理信息"""
        try:
            start_time = time.time()

            # 使用rasterio读取图像和地理信息
            image, profile = GeoTIFFHandler.read_geotiff(image_path)
            if image is None or profile is None:
                return False, 0, 0, 0, 0

            # 检测红色区域（不进行像素标准化）
            red_mask = RedRegionCleaner.detect_red_regions(image)
            total_red_pixels = np.sum(red_mask > 0)

            if total_red_pixels == 0:
                # 没有红色像素，直接保存
                success = GeoTIFFHandler.write_geotiff(output_path, image, profile)
                return success, 0, 0, 0, time.time() - start_time

            # 使用边界框优化：先找到红色轮廓
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            total_contours = len(contours)

            if total_contours == 0:
                success = GeoTIFFHandler.write_geotiff(output_path, image, profile)
                return success, 0, 0, 0, time.time() - start_time

            if progress_queue:
                try:
                    progress_queue.put({
                        'type': 'contour_found',
                        'process_id': process_id,
                        'count': total_contours,
                        'tool': 'cleaner'
                    })
                except:
                    pass

            # 对每个轮廓进行边界框优化的连通域清理
            result_image, stats = RedRegionCleaner.process_contours_with_bbox_optimization(
                image, red_mask, contours, min_area_threshold, white_color,
                progress_queue, process_id
            )

            # 使用rasterio保存结果，保留地理信息
            success = GeoTIFFHandler.write_geotiff(output_path, result_image, profile)

            # 发送完成统计
            if progress_queue:
                try:
                    progress_queue.put({
                        'type': 'cleaning_complete',
                        'process_id': process_id,
                        'stats': stats,
                        'tool': 'cleaner'
                    })
                except:
                    pass

            processing_time = time.time() - start_time

            return (success, stats['removed_count'], stats['removed_pixels'],
                    stats['kept_count'], processing_time)

        except Exception as e:
            print(f"进程 {process_id} 处理文件 {image_path} 时出错: {str(e)}")
            return False, 0, 0, 0, 0


class RedPixelDifferenceAnalyzer:
    """红色像素差值分析核心类 - 使用rasterio保留地理信息"""

    @staticmethod
    def extract_red_pixels(image_path):
        """提取图像中的红色像素掩膜"""
        try:
            image, profile = GeoTIFFHandler.read_geotiff(image_path)
            if image is None or profile is None:
                return None, None, None

            # 转换为HSV进行红色检测
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # 红色的HSV范围
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)

            return image, red_mask, profile

        except Exception as e:
            print(f"提取红色像素时出错: {str(e)}")
            return None, None, None

    @staticmethod
    def compute_red_pixel_difference(image1_path, image2_path, output_path, nodata_value=0, debug=False):
        """
        计算两张图像的红色像素差值 - 使用rasterio保留地理信息

        Args:
            image1_path: 第一张图像路径
            image2_path: 第二张图像路径
            output_path: 输出路径
            nodata_value: nodata值设置 (0=透明, 1=黑色, 2=白色)
            debug: 是否输出调试信息

        Returns:
            (是否成功, 结果统计)
        """
        try:
            start_time = time.time()

            if debug:
                print(f"处理文件对:")
                print(f"  图像1: {image1_path}")
                print(f"  图像2: {image2_path}")
                print(f"  输出: {output_path}")

            # 提取两张图像的红色像素
            image1, red_mask1, profile1 = RedPixelDifferenceAnalyzer.extract_red_pixels(image1_path)
            image2, red_mask2, profile2 = RedPixelDifferenceAnalyzer.extract_red_pixels(image2_path)

            if image1 is None or image2 is None or profile1 is None:
                if debug:
                    print(f"  错误: 无法读取图像文件")
                return False, None

            if debug:
                print(f"  图像1尺寸: {image1.shape}")
                print(f"  图像2尺寸: {image2.shape}")

            # 使用第一张图像的地理信息作为输出的地理信息
            output_profile = profile1.copy()

            # 确保图像尺寸一致
            if image1.shape != image2.shape:
                if debug:
                    print(f"  调整图像2尺寸: {image2.shape} -> {image1.shape}")
                # 将第二张图像调整为与第一张相同尺寸
                image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))
                red_mask2 = cv2.resize(red_mask2, (image1.shape[1], image1.shape[0]))

            # 计算差值掩膜：填充清理后有红色但种子生长前没有红色的区域（新增的红色区域）
            red_mask1_bool = red_mask1 > 0  # 种子生长前的红色掩膜
            red_mask2_bool = red_mask2 > 0  # 填充清理后的红色掩膜

            # 差值掩膜：在图像2（填充清理后）中有红色，但在图像1（种子生长前）中没有红色的区域
            difference_mask = red_mask2_bool & (~red_mask1_bool)

            # 统计信息
            image1_red_pixels = np.sum(red_mask1_bool)  # 种子生长前红色像素
            image2_red_pixels = np.sum(red_mask2_bool)  # 填充清理后红色像素
            difference_pixels = np.sum(difference_mask)  # 新增红色像素
            overlap_pixels = np.sum(red_mask1_bool & red_mask2_bool)  # 重叠红色像素

            if debug:
                print(f"  种子生长前红色像素: {image1_red_pixels:,}")
                print(f"  填充清理后红色像素: {image2_red_pixels:,}")
                print(f"  新增红色像素: {difference_pixels:,}")
                print(f"  重叠红色像素: {overlap_pixels:,}")

            # 创建结果图像
            if nodata_value == 0:  # 透明背景
                result_image = np.zeros((image1.shape[0], image1.shape[1], 4), dtype=np.uint8)  # BGRA
                # 在差值区域设置红色，alpha=255
                result_image[difference_mask] = [0, 0, 255, 255]  # BGRA: 蓝=0, 绿=0, 红=255, Alpha=255
                # 其他区域保持透明 (alpha=0)
                output_profile.update({'count': 4})

                if debug:
                    non_transparent_pixels = np.sum(result_image[:, :, 3] > 0)
                    print(f"  结果图像非透明像素: {non_transparent_pixels:,}")

            elif nodata_value == 1:  # 黑色背景
                result_image = np.zeros((image1.shape[0], image1.shape[1], 3), dtype=np.uint8)  # BGR
                result_image[difference_mask] = [0, 0, 255]  # BGR: 红色
                output_profile.update({'count': 3})

                if debug:
                    red_pixels_in_result = np.sum((result_image[:, :, 2] == 255) &
                                                  (result_image[:, :, 1] == 0) &
                                                  (result_image[:, :, 0] == 0))
                    print(f"  结果图像红色像素: {red_pixels_in_result:,}")

            else:  # 白色背景
                result_image = np.full((image1.shape[0], image1.shape[1], 3), 255, dtype=np.uint8)  # BGR
                result_image[difference_mask] = [0, 0, 255]  # BGR: 红色
                output_profile.update({'count': 3})

                if debug:
                    red_pixels_in_result = np.sum((result_image[:, :, 2] == 255) &
                                                  (result_image[:, :, 1] == 0) &
                                                  (result_image[:, :, 0] == 0))
                    print(f"  结果图像红色像素: {red_pixels_in_result:,}")

            # 使用rasterio保存结果，保留地理信息
            success = GeoTIFFHandler.write_geotiff(output_path, result_image, output_profile)

            processing_time = time.time() - start_time

            if debug:
                print(f"  保存成功: {success}")
                print(f"  处理时间: {processing_time:.2f}秒")

            stats = {
                'image1_red_pixels': image1_red_pixels,
                'image2_red_pixels': image2_red_pixels,
                'difference_pixels': difference_pixels,
                'overlap_pixels': overlap_pixels,
                'processing_time': processing_time
            }

            return success, stats

        except Exception as e:
            if debug:
                print(f"  处理出错: {str(e)}")
            print(f"计算红色像素差值时出错: {str(e)}")
            return False, None

    @staticmethod
    def batch_difference_analysis(folder1_path, folder2_path, output_folder, nodata_value=0,
                                  progress_callback=None, debug=False):
        """
        批量红色差值分析

        Args:
            folder1_path: 种子生长前影像文件夹路径
            folder2_path: 填充清理后影像文件夹路径
            output_folder: 输出文件夹路径
            nodata_value: 背景设置
            progress_callback: 进度回调函数
            debug: 是否启用调试模式

        Returns:
            (成功数量, 失败数量, 详细结果列表)
        """
        try:
            if debug:
                print(f"开始批量差值分析:")
                print(f"  文件夹1: {folder1_path}")
                print(f"  文件夹2: {folder2_path}")
                print(f"  输出文件夹: {output_folder}")

            # 获取文件夹1中的所有tif文件
            folder1_files = {}
            for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
                for file_path in Path(folder1_path).glob(ext):
                    # 提取文件名中的XXX_XXX模式
                    filename = file_path.stem
                    folder1_files[filename] = file_path

            if debug:
                print(f"  文件夹1中找到 {len(folder1_files)} 个文件:")
                for filename in list(folder1_files.keys())[:5]:  # 只显示前5个
                    print(f"    {filename}")
                if len(folder1_files) > 5:
                    print(f"    ... 还有 {len(folder1_files) - 5} 个文件")

            # 获取文件夹2中的所有tif文件
            folder2_files = {}
            for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
                for file_path in Path(folder2_path).glob(ext):
                    filename = file_path.stem
                    # 处理清理后的文件名，去掉前缀
                    if filename.startswith('cleaned_filled_'):
                        base_name = filename.replace('cleaned_filled_', '')
                        folder2_files[base_name] = file_path
                        if debug and len(folder2_files) <= 5:
                            print(f"    匹配: {filename} -> {base_name}")
                    else:
                        folder2_files[filename] = file_path

            if debug:
                print(f"  文件夹2中找到 {len(folder2_files)} 个文件:")
                for base_name in list(folder2_files.keys())[:5]:  # 只显示前5个
                    print(f"    {base_name}")
                if len(folder2_files) > 5:
                    print(f"    ... 还有 {len(folder2_files) - 5} 个文件")

            # 找到匹配的文件对
            matched_pairs = []
            for base_name in folder1_files:
                if base_name in folder2_files:
                    matched_pairs.append((
                        base_name,
                        folder1_files[base_name],
                        folder2_files[base_name]
                    ))

            if debug:
                print(f"  找到 {len(matched_pairs)} 个匹配的文件对")
                for base_name, file1, file2 in matched_pairs[:3]:  # 只显示前3个
                    print(f"    {base_name}: {file1.name} <-> {file2.name}")

            if not matched_pairs:
                if debug:
                    print("  没有找到匹配的文件对!")
                return 0, 0, []

            # 创建输出文件夹
            os.makedirs(output_folder, exist_ok=True)

            # 处理每个文件对
            success_count = 0
            failure_count = 0
            results = []

            total_pairs = len(matched_pairs)

            for i, (base_name, file1_path, file2_path) in enumerate(matched_pairs):
                try:
                    # 构建输出文件路径
                    output_filename = f"diff_{base_name}.tif"
                    output_path = os.path.join(output_folder, output_filename)

                    # 调用进度回调
                    if progress_callback:
                        progress_callback(i + 1, total_pairs, base_name, "处理中...")

                    if debug:
                        print(f"\n处理第 {i + 1}/{total_pairs} 个文件对: {base_name}")

                    # 执行差值分析（启用调试模式）
                    success, stats = RedPixelDifferenceAnalyzer.compute_red_pixel_difference(
                        str(file1_path), str(file2_path), output_path, nodata_value, debug=debug
                    )

                    if success and stats:
                        success_count += 1
                        result_info = {
                            'base_name': base_name,
                            'file1': str(file1_path),
                            'file2': str(file2_path),
                            'output': output_path,
                            'success': True,
                            'stats': stats
                        }
                        results.append(result_info)

                        # 更新进度
                        if progress_callback:
                            progress_callback(i + 1, total_pairs, base_name,
                                              f"成功 - 差值像素: {stats['difference_pixels']:,}")
                    else:
                        failure_count += 1
                        result_info = {
                            'base_name': base_name,
                            'file1': str(file1_path),
                            'file2': str(file2_path),
                            'output': output_path,
                            'success': False,
                            'error': '处理失败'
                        }
                        results.append(result_info)

                        # 更新进度
                        if progress_callback:
                            progress_callback(i + 1, total_pairs, base_name, "失败")

                except Exception as e:
                    failure_count += 1
                    result_info = {
                        'base_name': base_name,
                        'file1': str(file1_path),
                        'file2': str(file2_path),
                        'output': '',
                        'success': False,
                        'error': str(e)
                    }
                    results.append(result_info)

                    if debug:
                        print(f"  处理文件对 {base_name} 时出错: {str(e)}")

                    # 更新进度
                    if progress_callback:
                        progress_callback(i + 1, total_pairs, base_name, f"错误: {str(e)}")

            if debug:
                print(f"\n批量处理完成:")
                print(f"  成功: {success_count}")
                print(f"  失败: {failure_count}")

            return success_count, failure_count, results

        except Exception as e:
            if debug:
                print(f"批量差值分析时出错: {str(e)}")
            print(f"批量差值分析时出错: {str(e)}")
            return 0, 0, []


def worker_process_filler(process_id, file_list, output_folder, params, progress_queue):
    """轮廓填充的工作进程函数"""
    try:
        total_files = len(file_list)

        for i, tif_file in enumerate(file_list):
            # 发送文件开始处理信号
            progress_queue.put({
                'type': 'file_start',
                'process_id': process_id,
                'file_index': i + 1,
                'total_files': total_files,
                'filename': tif_file.name,
                'tool': 'filler'
            })

            # 构建输出路径
            output_path = os.path.join(output_folder, f"filled_{tif_file.name}")

            # 处理文件
            success, filled_pixels, gaps_analyzed, process_time, contour_count = ContourProcessor.detect_and_fill_internal_gaps(
                str(tif_file), output_path,
                progress_queue=progress_queue,
                process_id=process_id,
                **params
            )

            # 发送文件完成信号
            progress_queue.put({
                'type': 'file_complete',
                'process_id': process_id,
                'success': success,
                'filled_pixels': filled_pixels,
                'gaps_analyzed': gaps_analyzed,
                'contour_count': contour_count,
                'process_time': process_time,
                'tool': 'filler'
            })

        # 发送进程完成信号
        progress_queue.put({
            'type': 'process_complete',
            'process_id': process_id,
            'tool': 'filler'
        })

    except Exception as e:
        progress_queue.put({
            'type': 'process_error',
            'process_id': process_id,
            'error': str(e),
            'tool': 'filler'
        })


def worker_process_cleaner(process_id, file_list, output_folder, params, progress_queue):
    """红色连通域清理的工作进程函数"""
    try:
        total_files = len(file_list)

        for i, tif_file in enumerate(file_list):
            # 发送文件开始处理信号
            progress_queue.put({
                'type': 'file_start',
                'process_id': process_id,
                'file_index': i + 1,
                'total_files': total_files,
                'filename': tif_file.name,
                'tool': 'cleaner'
            })

            # 构建输出路径
            output_path = os.path.join(output_folder, f"cleaned_{tif_file.name}")

            # 处理文件
            success, removed_count, removed_pixels, kept_count, process_time = RedRegionCleaner.clean_small_red_regions(
                str(tif_file), output_path,
                progress_queue=progress_queue,
                process_id=process_id,
                **params
            )

            # 发送文件完成信号
            progress_queue.put({
                'type': 'file_complete',
                'process_id': process_id,
                'success': success,
                'removed_count': removed_count,
                'removed_pixels': removed_pixels,
                'kept_count': kept_count,
                'process_time': process_time,
                'tool': 'cleaner'
            })

        # 发送进程完成信号
        progress_queue.put({
            'type': 'process_complete',
            'process_id': process_id,
            'tool': 'cleaner'
        })

    except Exception as e:
        progress_queue.put({
            'type': 'process_error',
            'process_id': process_id,
            'error': str(e),
            'tool': 'cleaner'
        })


class ProgressMonitorThread(QThread):
    """通用进度监控线程"""
    progress_updated = pyqtSignal(int, str, str, int, int, str)  # process_id, file_name, status, current, total, tool
    log_message = pyqtSignal(str, str)  # message, tool
    all_complete = pyqtSignal(str, str)  # summary, tool

    def __init__(self, progress_queue, num_processes, tool_name):
        super().__init__()
        self.progress_queue = progress_queue
        self.num_processes = num_processes
        self.tool_name = tool_name
        self.is_running = True
        self.completed_processes = 0

        # 统计信息
        self.total_files_processed = 0
        self.total_success = 0

        # 工具特定统计
        if tool_name == 'filler':
            self.total_filled_pixels = 0
            self.total_gaps_analyzed = 0
            self.total_contours = 0
        else:  # cleaner
            self.total_removed_regions = 0
            self.total_removed_pixels = 0
            self.total_kept_regions = 0

    def stop(self):
        self.is_running = False

    def run(self):
        while self.is_running and self.completed_processes < self.num_processes:
            try:
                msg = self.progress_queue.get(timeout=1)

                # 只处理当前工具的消息
                if msg.get('tool') != self.tool_name:
                    continue

                if msg['type'] == 'file_start':
                    status = f"处理文件 {msg['file_index']}/{msg['total_files']}: {msg['filename']}"
                    self.progress_updated.emit(msg['process_id'], msg['filename'], status, 0, 100, self.tool_name)
                    self.log_message.emit(f"进程 {msg['process_id'] + 1}: {status}", self.tool_name)

                elif msg['type'] == 'preprocessing':
                    if self.tool_name == 'filler':  # 只有填充工具需要预处理日志
                        self.log_message.emit(f"进程 {msg['process_id'] + 1}: {msg['message']}", self.tool_name)

                elif msg['type'] == 'preprocessing_complete':
                    if self.tool_name == 'filler':  # 只有填充工具需要预处理日志
                        self.log_message.emit(
                            f"进程 {msg['process_id'] + 1}: 标准化了 {msg['standardized_pixels']:,} 个红色像素",
                            self.tool_name)

                elif msg['type'] == 'contour_found':
                    if self.tool_name == 'filler':
                        self.log_message.emit(f"进程 {msg['process_id'] + 1}: 发现 {msg['count']} 个轮廓",
                                              self.tool_name)
                    else:  # cleaner
                        self.log_message.emit(f"进程 {msg['process_id'] + 1}: 发现 {msg['count']} 个红色轮廓区域",
                                              self.tool_name)

                elif msg['type'] == 'contour_filtered':
                    if self.tool_name == 'filler':  # 只有填充工具使用轮廓过滤
                        self.log_message.emit(
                            f"进程 {msg['process_id'] + 1}: 过滤轮廓 {msg['original']} → {msg['filtered']} (保留 {msg['filtered'] / msg['original'] * 100:.1f}%)",
                            self.tool_name)

                elif msg['type'] == 'contour_progress':
                    if self.tool_name == 'filler':  # 只有填充工具使用轮廓进度
                        current = msg['current']
                        total = msg['total']
                        progress = int((current / total) * 100) if total > 0 else 0
                        status = f"轮廓进度: {current}/{total}"
                        self.progress_updated.emit(msg['process_id'], "", status, progress, 100, self.tool_name)

                elif msg['type'] == 'analyzing_components':
                    if 'total_red_pixels' in msg:
                        self.log_message.emit(
                            f"进程 {msg['process_id'] + 1}: 分析 {msg['total_red_pixels']:,} 个红色像素的连通域",
                            self.tool_name)
                    elif 'message' in msg:
                        self.log_message.emit(f"进程 {msg['process_id'] + 1}: {msg['message']}", self.tool_name)

                elif msg['type'] == 'components_found':
                    self.log_message.emit(f"进程 {msg['process_id'] + 1}: 发现 {msg['count']} 个真正的连通域",
                                          self.tool_name)
                    status = f"分析连通域: {msg['count']} 个"
                    self.progress_updated.emit(msg['process_id'], "", status, 50, 100, self.tool_name)

                elif msg['type'] == 'component_progress':
                    current = msg['current']
                    total = msg['total']
                    progress = int((current / total) * 100) if total > 0 else 0
                    status = f"连通域进度: {current}/{total}"
                    self.progress_updated.emit(msg['process_id'], "", status, progress, 100, self.tool_name)

                elif msg['type'] == 'cleaning_complete':
                    stats = msg['stats']
                    self.log_message.emit(
                        f"进程 {msg['process_id'] + 1}: 删除 {stats['removed_count']} 个小区域({stats['removed_pixels']} 像素), 保留 {stats['kept_count']} 个大区域",
                        self.tool_name)

                elif msg['type'] == 'file_complete':
                    self.total_files_processed += 1
                    if msg['success']:
                        self.total_success += 1
                        if self.tool_name == 'filler':
                            self.total_filled_pixels += msg['filled_pixels']
                            self.total_gaps_analyzed += msg['gaps_analyzed']
                            self.total_contours += msg['contour_count']
                            self.log_message.emit(
                                f"进程 {msg['process_id'] + 1}: 文件处理完成，填充 {msg['filled_pixels']} 像素",
                                self.tool_name)
                        else:  # cleaner
                            self.total_removed_regions += msg['removed_count']
                            self.total_removed_pixels += msg['removed_pixels']
                            self.total_kept_regions += msg['kept_count']
                            self.log_message.emit(
                                f"进程 {msg['process_id'] + 1}: 文件处理完成，删除 {msg['removed_count']} 个区域",
                                self.tool_name)

                    status = f"完成 - 耗时 {msg['process_time']:.1f}s"
                    self.progress_updated.emit(msg['process_id'], "", status, 100, 100, self.tool_name)

                elif msg['type'] == 'process_complete':
                    self.completed_processes += 1
                    self.log_message.emit(f"进程 {msg['process_id'] + 1} 已完成所有任务", self.tool_name)

                elif msg['type'] == 'process_error':
                    self.completed_processes += 1
                    self.log_message.emit(f"进程 {msg['process_id'] + 1} 出现错误: {msg['error']}", self.tool_name)

            except queue.Empty:
                continue
            except Exception as e:
                self.log_message.emit(f"监控线程错误: {str(e)}", self.tool_name)
                break

        # 发送完成总结
        if self.tool_name == 'filler':
            summary = f"""8进程轮廓填充完成！(保留地理信息)
总处理文件: {self.total_files_processed}
成功处理: {self.total_success}
总处理轮廓: {self.total_contours}
总填充像素: {self.total_filled_pixels:,}
总分析空隙: {self.total_gaps_analyzed}"""
        else:  # cleaner
            summary = f"""8进程红色连通域清理完成！(保留地理信息)
总处理文件: {self.total_files_processed}
成功处理: {self.total_success}
总删除区域: {self.total_removed_regions}
总删除像素: {self.total_removed_pixels:,}
总保留区域: {self.total_kept_regions}"""

        self.all_complete.emit(summary, self.tool_name)


class ProcessProgressWidget(QWidget):
    """通用进程进度显示组件"""

    def __init__(self, process_id, tool_name):
        super().__init__()
        self.process_id = process_id
        self.tool_name = tool_name
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Process title
        tool_display = "Fill" if self.tool_name == "filler" else "Clean"
        self.title_label = QLabel(f"{tool_display} Process {self.process_id + 1}")
        self.title_label.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(self.title_label)

        # Current file label
        self.file_label = QLabel("Waiting for task assignment...")
        self.file_label.setStyleSheet("color: #666; font-size: 9px;")
        layout.addWidget(self.file_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setFormat("%p%")
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Idle")
        self.status_label.setStyleSheet("color: #888; font-size: 8px;")
        layout.addWidget(self.status_label)

        # 设置边框样式
        self.setStyleSheet("""
            ProcessProgressWidget {
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: #fafafa;
            }
        """)

    def update_progress(self, filename, status, progress):
        """Update progress display"""
        if filename:
            self.file_label.setText(f"File: {filename}")
        self.status_label.setText(status)
        self.progress_bar.setValue(progress)

        # Change color based on status
        if "Complete" in status or "完成" in status:
            color = "#4CAF50"  # Green
        elif "Contour" in status or "轮廓进度" in status:
            color = "#2196F3"  # Blue
        elif "Analyzing" in status or "分析" in status:
            color = "#FF9800"  # Orange
        else:
            color = None

        if color:
            self.progress_bar.setStyleSheet(f"""
                QProgressBar::chunk {{
                    background-color: {color};
                }}
            """)
        else:
            self.progress_bar.setStyleSheet("")


class DualToolImageProcessorGUI(QMainWindow):
    """双工具图像处理GUI - 轮廓填充和红色连通域清理 - 保留地理信息版本"""

    def __init__(self):
        super().__init__()
        self.processes = {'filler': [], 'cleaner': []}
        self.progress_queues = {'filler': None, 'cleaner': None}
        self.monitor_threads = {'filler': None, 'cleaner': None}
        self.init_ui()

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("8-Process GeoTIFF Processing Tool - Contour Fill & Connected Component Cleanup & Red Difference Analysis (Preserve Geographic Info)")
        self.setGeometry(100, 100, 1400, 900)

        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom: 2px solid #2196F3;
            }
            QTabBar::tab:hover {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                opacity: 0.8;
            }
            QPushButton:pressed {
                opacity: 0.6;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
        """)

        # 创建中央widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 创建主布局
        main_layout = QVBoxLayout(central_widget)

        # 创建标签页控件
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # 创建三个标签页
        self.create_filler_tab()
        self.create_cleaner_tab()
        self.create_difference_tab()

        # 底部状态栏
        self.create_status_bar()

    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("8-Process GeoTIFF Processing Tool Started - Using rasterio to preserve geographic information")

    def create_filler_tab(self):
        """创建轮廓填充标签页"""
        filler_widget = QWidget()
        self.tab_widget.addTab(filler_widget, "Contour Fill Tool")

        # 创建布局
        layout = QHBoxLayout(filler_widget)

        # 左侧控制面板
        left_panel = self.create_filler_control_panel()
        layout.addWidget(left_panel, 1)

        # 中间进程监控面板
        middle_panel = self.create_filler_process_panel()
        layout.addWidget(middle_panel, 2)

        # 右侧日志面板
        right_panel = self.create_filler_log_panel()
        layout.addWidget(right_panel, 1)

    def create_cleaner_tab(self):
        """创建连通域清理标签页"""
        cleaner_widget = QWidget()
        self.tab_widget.addTab(cleaner_widget, "Connected Component Cleanup Tool")

        # 创建布局
        layout = QHBoxLayout(cleaner_widget)

        # 左侧控制面板
        left_panel = self.create_cleaner_control_panel()
        layout.addWidget(left_panel, 1)

        # 中间进程监控面板
        middle_panel = self.create_cleaner_process_panel()
        layout.addWidget(middle_panel, 2)

        # 右侧日志面板
        right_panel = self.create_cleaner_log_panel()
        layout.addWidget(right_panel, 1)

    def create_difference_tab(self):
        """创建红色差值分析标签页"""
        difference_widget = QWidget()
        self.tab_widget.addTab(difference_widget, "Red Difference Analysis")

        # 创建布局
        layout = QHBoxLayout(difference_widget)

        # 左侧控制面板
        left_panel = self.create_difference_control_panel()
        layout.addWidget(left_panel, 1)

        # 右侧日志面板
        right_panel = self.create_difference_log_panel()
        layout.addWidget(right_panel, 1)

    def create_filler_control_panel(self):
        """创建轮廓填充控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 文件夹选择组
        folder_group = QGroupBox("GeoTIFF Folder Settings")
        folder_layout = QGridLayout(folder_group)

        # Input folder
        folder_layout.addWidget(QLabel("Input Folder:"), 0, 0)
        self.filler_input_folder_edit = QLineEdit()
        self.filler_input_folder_edit.setPlaceholderText("Select folder containing GeoTIFF files")
        folder_layout.addWidget(self.filler_input_folder_edit, 1, 0, 1, 2)
        filler_input_btn = QPushButton("Browse")
        filler_input_btn.setStyleSheet("QPushButton { background-color: #4CAF50; }")
        filler_input_btn.clicked.connect(lambda: self.select_folder(self.filler_input_folder_edit, "Input"))
        folder_layout.addWidget(filler_input_btn, 1, 2)

        # Output folder
        folder_layout.addWidget(QLabel("Output Folder:"), 2, 0)
        self.filler_output_folder_edit = QLineEdit()
        self.filler_output_folder_edit.setPlaceholderText("Select output folder")
        folder_layout.addWidget(self.filler_output_folder_edit, 3, 0, 1, 2)
        filler_output_btn = QPushButton("Browse")
        filler_output_btn.setStyleSheet("QPushButton { background-color: #4CAF50; }")
        filler_output_btn.clicked.connect(lambda: self.select_folder(self.filler_output_folder_edit, "Output"))
        folder_layout.addWidget(filler_output_btn, 3, 2)

        layout.addWidget(folder_group)

        # Fill parameters settings group
        params_group = QGroupBox("Fill Parameters")
        params_layout = QGridLayout(params_group)

        # Mean threshold
        params_layout.addWidget(QLabel("Mean Threshold:"), 0, 0)
        self.filler_mean_threshold_spin = QSpinBox()
        self.filler_mean_threshold_spin.setRange(1, 255)
        self.filler_mean_threshold_spin.setValue(95)
        params_layout.addWidget(self.filler_mean_threshold_spin, 0, 1)

        # Standard deviation threshold
        params_layout.addWidget(QLabel("Std Threshold:"), 1, 0)
        self.filler_std_threshold_spin = QSpinBox()
        self.filler_std_threshold_spin.setRange(1, 255)
        self.filler_std_threshold_spin.setValue(50)
        params_layout.addWidget(self.filler_std_threshold_spin, 1, 1)

        # Small gap threshold
        params_layout.addWidget(QLabel("Small Gap Threshold:"), 2, 0)
        self.filler_small_gap_spin = QSpinBox()
        self.filler_small_gap_spin.setRange(1, 1000)
        self.filler_small_gap_spin.setValue(50)
        params_layout.addWidget(self.filler_small_gap_spin, 2, 1)

        # Batch size
        params_layout.addWidget(QLabel("Batch Size:"), 3, 0)
        self.filler_batch_size_spin = QSpinBox()
        self.filler_batch_size_spin.setRange(1, 200)
        self.filler_batch_size_spin.setValue(16)
        params_layout.addWidget(self.filler_batch_size_spin, 3, 1)

        # Fill color option
        params_layout.addWidget(QLabel("Fill Mode:"), 4, 0)
        self.filler_fill_mode_combo = QComboBox()
        self.filler_fill_mode_combo.addItems(["Red Fill", "Green Fill (Debug)"])
        params_layout.addWidget(self.filler_fill_mode_combo, 4, 1)

        layout.addWidget(params_group)

        # 地理信息提示组
        geo_info_group = QGroupBox("地理信息处理")
        geo_info_layout = QVBoxLayout(geo_info_group)

        geo_info_label = QLabel("""
<b>✓ 使用rasterio保留地理信息</b><br>
<b>✓ 保持坐标系统 (CRS)</b><br>
<b>✓ 保持地理变换 (Transform)</b><br>
<b>✓ 保持元数据 (Metadata)</b><br>
<font color="red">⚠ 确保输入文件为GeoTIFF格式</font>
        """)
        geo_info_label.setStyleSheet("color: #444; font-size: 10px;")
        geo_info_layout.addWidget(geo_info_label)

        layout.addWidget(geo_info_group)

        # Control buttons group
        control_group = QGroupBox("Operation Control")
        control_layout = QVBoxLayout(control_group)

        # Start processing button
        self.filler_start_btn = QPushButton("Start 8-Process Contour Fill")
        self.filler_start_btn.setStyleSheet("QPushButton { background-color: #4CAF50; }")
        self.filler_start_btn.clicked.connect(lambda: self.start_processing('filler'))
        control_layout.addWidget(self.filler_start_btn)

        # Stop processing button
        self.filler_stop_btn = QPushButton("Stop All Processes")
        self.filler_stop_btn.setStyleSheet("QPushButton { background-color: #f44336; }")
        self.filler_stop_btn.clicked.connect(lambda: self.stop_processing('filler'))
        self.filler_stop_btn.setEnabled(False)
        control_layout.addWidget(self.filler_stop_btn)

        layout.addWidget(control_group)
        layout.addStretch()

        return panel

    def create_cleaner_control_panel(self):
        """创建连通域清理控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 文件夹选择组
        folder_group = QGroupBox("GeoTIFF Folder Settings")
        folder_layout = QGridLayout(folder_group)

        # Input folder
        folder_layout.addWidget(QLabel("Input Folder:"), 0, 0)
        self.cleaner_input_folder_edit = QLineEdit()
        self.cleaner_input_folder_edit.setPlaceholderText("Select folder containing GeoTIFF files")
        folder_layout.addWidget(self.cleaner_input_folder_edit, 1, 0, 1, 2)
        cleaner_input_btn = QPushButton("Browse")
        cleaner_input_btn.setStyleSheet("QPushButton { background-color: #FF5722; }")
        cleaner_input_btn.clicked.connect(lambda: self.select_folder(self.cleaner_input_folder_edit, "Input"))
        folder_layout.addWidget(cleaner_input_btn, 1, 2)

        # Output folder
        folder_layout.addWidget(QLabel("Output Folder:"), 2, 0)
        self.cleaner_output_folder_edit = QLineEdit()
        self.cleaner_output_folder_edit.setPlaceholderText("Select output folder")
        folder_layout.addWidget(self.cleaner_output_folder_edit, 3, 0, 1, 2)
        cleaner_output_btn = QPushButton("Browse")
        cleaner_output_btn.setStyleSheet("QPushButton { background-color: #FF5722; }")
        cleaner_output_btn.clicked.connect(lambda: self.select_folder(self.cleaner_output_folder_edit, "Output"))
        folder_layout.addWidget(cleaner_output_btn, 3, 2)

        layout.addWidget(folder_group)

        # Cleanup parameters settings group
        params_group = QGroupBox("Cleanup Parameters")
        params_layout = QGridLayout(params_group)

        # Minimum area threshold
        params_layout.addWidget(QLabel("Min Area Threshold:"), 0, 0)
        self.cleaner_min_area_spin = QSpinBox()
        self.cleaner_min_area_spin.setRange(1, 100000)
        self.cleaner_min_area_spin.setValue(500)
        self.cleaner_min_area_spin.setSuffix(" pixels")
        params_layout.addWidget(self.cleaner_min_area_spin, 0, 1)

        # Replacement color settings
        params_layout.addWidget(QLabel("Replace Color:"), 1, 0)
        self.cleaner_replace_color_combo = QComboBox()
        self.cleaner_replace_color_combo.addItems(["White (255,255,255)", "Black (0,0,0)", "Gray (128,128,128)"])
        params_layout.addWidget(self.cleaner_replace_color_combo, 1, 1)

        layout.addWidget(params_group)

        # Geographic information hint group
        geo_info_group = QGroupBox("Geographic Information Processing")
        geo_info_layout = QVBoxLayout(geo_info_group)

        geo_info_label = QLabel("""
<b>✓ Use rasterio to preserve geographic information</b><br>
<b>✓ Maintain coordinate system (CRS)</b><br>
<b>✓ Maintain geotransform (Transform)</b><br>
<b>✓ Maintain metadata (Metadata)</b><br>
<font color="red">⚠ Ensure input files are in GeoTIFF format</font>
        """)
        geo_info_label.setStyleSheet("color: #444; font-size: 10px;")
        geo_info_layout.addWidget(geo_info_label)

        layout.addWidget(geo_info_group)

        # Statistics information group
        stats_group = QGroupBox("Operation Instructions")
        stats_layout = QVBoxLayout(stats_group)

        preview_label = QLabel("""
<b>Cleanup Process:</b><br>
• Detect all red connected components<br>
• Calculate area of each connected component<br>
• Remove connected components smaller than threshold<br>
• Replace removed regions with specified color<br>
• Keep connected components larger than threshold
        """)
        preview_label.setStyleSheet("color: #444; font-size: 10px;")
        stats_layout.addWidget(preview_label)

        layout.addWidget(stats_group)

        # Control buttons group
        control_group = QGroupBox("Operation Control")
        control_layout = QVBoxLayout(control_group)

        # Start processing button
        self.cleaner_start_btn = QPushButton("Start 8-Process Connected Component Cleanup")
        self.cleaner_start_btn.setStyleSheet("QPushButton { background-color: #FF5722; }")
        self.cleaner_start_btn.clicked.connect(lambda: self.start_processing('cleaner'))
        control_layout.addWidget(self.cleaner_start_btn)

        # Stop processing button
        self.cleaner_stop_btn = QPushButton("Stop All Processes")
        self.cleaner_stop_btn.setStyleSheet("QPushButton { background-color: #f44336; }")
        self.cleaner_stop_btn.clicked.connect(lambda: self.stop_processing('cleaner'))
        self.cleaner_stop_btn.setEnabled(False)
        control_layout.addWidget(self.cleaner_stop_btn)

        layout.addWidget(control_group)
        layout.addStretch()

        return panel

    def create_filler_process_panel(self):
        """创建轮廓填充进程监控面板"""
        panel = QGroupBox("8-Process Contour Fill Monitoring")
        main_layout = QVBoxLayout(panel)

        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QGridLayout(scroll_widget)

        self.filler_process_widgets = []
        for i in range(8):
            process_widget = ProcessProgressWidget(i, "filler")
            self.filler_process_widgets.append(process_widget)
            row = i // 2
            col = i % 2
            scroll_layout.addWidget(process_widget, row, col)

        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        main_layout.addWidget(scroll)

        return panel

    def create_cleaner_process_panel(self):
        """创建连通域清理进程监控面板"""
        panel = QGroupBox("8-Process Connected Component Cleanup Monitoring")
        main_layout = QVBoxLayout(panel)

        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_layout = QGridLayout(scroll_widget)

        self.cleaner_process_widgets = []
        for i in range(8):
            process_widget = ProcessProgressWidget(i, "cleaner")
            self.cleaner_process_widgets.append(process_widget)
            row = i // 2
            col = i % 2
            scroll_layout.addWidget(process_widget, row, col)

        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        main_layout.addWidget(scroll)

        return panel

    def create_filler_log_panel(self):
        """创建轮廓填充日志面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        log_label = QLabel("Contour Fill Log")
        log_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(log_label)

        self.filler_log_text = QTextEdit()
        self.filler_log_text.setReadOnly(True)
        self.filler_log_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.filler_log_text)

        clear_btn = QPushButton("Clear Log")
        clear_btn.setStyleSheet("QPushButton { background-color: #ff9800; }")
        clear_btn.clicked.connect(lambda: self.clear_log('filler'))
        layout.addWidget(clear_btn)

        return panel

    def create_cleaner_log_panel(self):
        """创建连通域清理日志面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        log_label = QLabel("Connected Component Cleanup Log")
        log_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(log_label)

        self.cleaner_log_text = QTextEdit()
        self.cleaner_log_text.setReadOnly(True)
        self.cleaner_log_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.cleaner_log_text)

        clear_btn = QPushButton("Clear Log")
        clear_btn.setStyleSheet("QPushButton { background-color: #ff9800; }")
        clear_btn.clicked.connect(lambda: self.clear_log('cleaner'))
        layout.addWidget(clear_btn)

        return panel

    def create_difference_control_panel(self):
        """创建红色差值分析控制面板 - 批量处理版本"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Folder selection group
        folder_group = QGroupBox("Batch GeoTIFF Folder Settings")
        folder_layout = QGridLayout(folder_group)

        # Seed growth before image folder
        folder_layout.addWidget(QLabel("Before Seed Growth Image Folder:"), 0, 0)
        self.diff_folder1_edit = QLineEdit()
        self.diff_folder1_edit.setPlaceholderText("Select folder containing before seed growth images (e.g.: 000_002.tif)")
        folder_layout.addWidget(self.diff_folder1_edit, 1, 0, 1, 2)
        diff_folder1_btn = QPushButton("Browse")
        diff_folder1_btn.setStyleSheet("QPushButton { background-color: #9C27B0; }")
        diff_folder1_btn.clicked.connect(lambda: self.select_diff_folder(self.diff_folder1_edit, "Before Seed Growth"))
        folder_layout.addWidget(diff_folder1_btn, 1, 2)

        # After fill cleanup image folder
        folder_layout.addWidget(QLabel("After Fill Cleanup Image Folder:"), 2, 0)
        self.diff_folder2_edit = QLineEdit()
        self.diff_folder2_edit.setPlaceholderText("Select folder containing after fill cleanup images (e.g.: cleaned_filled_000_002.tif)")
        folder_layout.addWidget(self.diff_folder2_edit, 3, 0, 1, 2)
        diff_folder2_btn = QPushButton("Browse")
        diff_folder2_btn.setStyleSheet("QPushButton { background-color: #9C27B0; }")
        diff_folder2_btn.clicked.connect(lambda: self.select_diff_folder(self.diff_folder2_edit, "After Fill Cleanup"))
        folder_layout.addWidget(diff_folder2_btn, 3, 2)

        # Output folder
        folder_layout.addWidget(QLabel("Output Folder:"), 4, 0)
        self.diff_output_folder_edit = QLineEdit()
        self.diff_output_folder_edit.setPlaceholderText("Select difference analysis result output folder")
        folder_layout.addWidget(self.diff_output_folder_edit, 5, 0, 1, 2)
        diff_output_folder_btn = QPushButton("Browse")
        diff_output_folder_btn.setStyleSheet("QPushButton { background-color: #9C27B0; }")
        diff_output_folder_btn.clicked.connect(lambda: self.select_diff_folder(self.diff_output_folder_edit, "Output"))
        folder_layout.addWidget(diff_output_folder_btn, 5, 2)

        layout.addWidget(folder_group)

        # Processing parameters settings group
        params_group = QGroupBox("Processing Parameters")
        params_layout = QGridLayout(params_group)

        # NoData settings
        params_layout.addWidget(QLabel("Background Setting:"), 0, 0)
        self.diff_nodata_combo = QComboBox()
        self.diff_nodata_combo.addItems(["Transparent Background", "Black Background", "White Background"])
        self.diff_nodata_combo.setToolTip("Select background color for non-difference regions")
        params_layout.addWidget(self.diff_nodata_combo, 0, 1)

        layout.addWidget(params_group)

        # Batch processing progress group
        progress_group = QGroupBox("Batch Processing Progress")
        progress_layout = QVBoxLayout(progress_group)

        # Progress bar
        self.diff_progress_bar = QProgressBar()
        self.diff_progress_bar.setValue(0)
        self.diff_progress_bar.setFormat("Waiting to start... (0/0)")
        progress_layout.addWidget(self.diff_progress_bar)

        # Current processing status
        self.diff_current_status = QLabel("Status: Waiting to start")
        self.diff_current_status.setStyleSheet("color: #666; font-size: 10px;")
        progress_layout.addWidget(self.diff_current_status)

        layout.addWidget(progress_group)

        # Geographic information hint group
        geo_info_group = QGroupBox("Batch Processing Instructions")
        geo_info_layout = QVBoxLayout(geo_info_group)

        geo_info_label = QLabel("""
<b>Batch File Matching Rules:</b><br>
• Before seed growth: XXX_XXX.tif (e.g.: 000_002.tif)<br>
• After fill cleanup: cleaned_filled_XXX_XXX.tif<br>
• Automatically match same XXX_XXX pattern<br>
• Output file: diff_XXX_XXX.tif<br><br>
<b>✓ Use rasterio to preserve geographic information</b><br>
<b>✓ Sequential processing to avoid system lag</b><br>
<font color="red">⚠ Ensure file name format is correct</font>
        """)
        geo_info_label.setStyleSheet("color: #444; font-size: 10px;")
        geo_info_layout.addWidget(geo_info_label)

        layout.addWidget(geo_info_group)

        # Algorithm description group
        info_group = QGroupBox("Difference Analysis Algorithm")
        info_layout = QVBoxLayout(info_group)

        info_label = QLabel("""
<b>Batch Difference Analysis Process:</b><br>
• Scan GeoTIFF files in both folders<br>
• Automatically match file pairs based on XXX_XXX pattern<br>
• Extract red pixels from each image pair<br>
• Calculate difference: Before seed growth - After fill cleanup<br>
• Keep red regions present before seed growth but not after fill cleanup<br>
• Batch output difference results, preserve geographic information<br><br>
<b>Application Scenarios:</b><br>
• Seed growth effect evaluation<br>
• Fill cleanup quality detection<br>
• Batch change detection analysis
        """)
        info_label.setStyleSheet("color: #444; font-size: 10px;")
        info_layout.addWidget(info_label)

        layout.addWidget(info_group)

        # Control buttons group
        control_group = QGroupBox("Batch Operation Control")
        control_layout = QVBoxLayout(control_group)

        # Preview file matches button
        self.diff_preview_btn = QPushButton("Preview File Match Results")
        self.diff_preview_btn.setStyleSheet("QPushButton { background-color: #607D8B; }")
        self.diff_preview_btn.clicked.connect(self.preview_file_matches)
        control_layout.addWidget(self.diff_preview_btn)

        # Start batch processing button
        self.diff_start_btn = QPushButton("Start Batch Red Difference Analysis")
        self.diff_start_btn.setStyleSheet("QPushButton { background-color: #9C27B0; }")
        self.diff_start_btn.clicked.connect(self.start_batch_difference_analysis)
        control_layout.addWidget(self.diff_start_btn)

        # Stop processing button
        self.diff_stop_btn = QPushButton("Stop Batch Processing")
        self.diff_stop_btn.setStyleSheet("QPushButton { background-color: #f44336; }")
        self.diff_stop_btn.clicked.connect(self.stop_batch_difference_analysis)
        self.diff_stop_btn.setEnabled(False)
        control_layout.addWidget(self.diff_stop_btn)

        layout.addWidget(control_group)
        layout.addStretch()

        return panel

    def create_difference_log_panel(self):
        """创建红色差值分析日志面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        log_label = QLabel("Difference Analysis Log")
        log_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(log_label)

        self.diff_log_text = QTextEdit()
        self.diff_log_text.setReadOnly(True)
        self.diff_log_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.diff_log_text)

        clear_btn = QPushButton("清除日志")
        clear_btn.setStyleSheet("QPushButton { background-color: #ff9800; }")
        clear_btn.clicked.connect(self.clear_difference_log)
        layout.addWidget(clear_btn)

        return panel

    def select_folder(self, line_edit, folder_type):
        """Select folder"""
        folder = QFileDialog.getExistingDirectory(self, f"Select {folder_type} Folder")
        if folder:
            line_edit.setText(folder)
            current_tab_index = self.tab_widget.currentIndex()
            tab_name = "Contour Fill" if current_tab_index == 0 else "Connected Component Cleanup"
            self.log_message(f"Selected {folder_type} folder: {folder}", 'filler' if current_tab_index == 0 else 'cleaner')

    def select_diff_folder(self, line_edit, folder_type):
        """Select difference analysis folder"""
        folder = QFileDialog.getExistingDirectory(self, f"Select {folder_type} Folder")
        if folder:
            line_edit.setText(folder)
            self.log_difference_message(f"Selected {folder_type} folder: {folder}")

    def preview_file_matches(self):
        """预览文件匹配结果"""
        folder1_path = self.diff_folder1_edit.text().strip()
        folder2_path = self.diff_folder2_edit.text().strip()

        if not folder1_path or not folder2_path:
            QMessageBox.warning(self, "Warning", "Please select two input folders first")
            return

        if not os.path.exists(folder1_path) or not os.path.exists(folder2_path):
            QMessageBox.warning(self, "Warning", "Selected folder does not exist")
            return

        try:
            # 获取文件夹1中的所有tif文件
            folder1_files = {}
            for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
                for file_path in Path(folder1_path).glob(ext):
                    filename = file_path.stem
                    folder1_files[filename] = file_path

            # 获取文件夹2中的所有tif文件
            folder2_files = {}
            for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
                for file_path in Path(folder2_path).glob(ext):
                    filename = file_path.stem
                    # 处理清理后的文件名，去掉前缀
                    if filename.startswith('cleaned_filled_'):
                        base_name = filename.replace('cleaned_filled_', '')
                        folder2_files[base_name] = file_path
                    else:
                        folder2_files[filename] = file_path

            # 找到匹配的文件对
            matched_pairs = []
            unmatched_files1 = []
            unmatched_files2 = []

            for base_name in folder1_files:
                if base_name in folder2_files:
                    matched_pairs.append((base_name, folder1_files[base_name].name, folder2_files[base_name].name))
                else:
                    unmatched_files1.append(folder1_files[base_name].name)

            for base_name in folder2_files:
                if base_name not in folder1_files:
                    unmatched_files2.append(folder2_files[base_name].name)

            # 生成预览信息
            preview_text = f"""文件匹配预览结果：

【匹配的文件对】({len(matched_pairs)} 对)：
"""
            for base_name, file1, file2 in matched_pairs:
                preview_text += f"• {base_name}: {file1} ↔ {file2}\n"

            if unmatched_files1:
                preview_text += f"\n【种子生长前文件夹中无匹配的文件】({len(unmatched_files1)} 个)：\n"
                for file in unmatched_files1:
                    preview_text += f"• {file}\n"

            if unmatched_files2:
                preview_text += f"\n【填充清理后文件夹中无匹配的文件】({len(unmatched_files2)} 个)：\n"
                for file in unmatched_files2:
                    preview_text += f"• {file}\n"

            preview_text += f"\n将处理 {len(matched_pairs)} 个文件对进行差值分析。"

            # Show preview dialog
            msg = QMessageBox()
            msg.setWindowTitle("File Match Preview")
            msg.setText("File Match Preview Results")
            msg.setDetailedText(preview_text)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

            # Log to log
            self.log_difference_message(f"Preview complete: Found {len(matched_pairs)} matched file pairs")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error previewing file matches:\n{str(e)}")
            self.log_difference_message(f"Error previewing file matches: {str(e)}")

    def start_batch_difference_analysis(self):
        """开始批量红色差值分析"""
        folder1_path = self.diff_folder1_edit.text().strip()
        folder2_path = self.diff_folder2_edit.text().strip()
        output_folder = self.diff_output_folder_edit.text().strip()

        # Validate input
        if not folder1_path or not folder2_path or not output_folder:
            QMessageBox.warning(self, "Warning", "Please select two input folders and output folder")
            return

        if not os.path.exists(folder1_path):
            QMessageBox.warning(self, "Warning", "Before seed growth image folder does not exist")
            return

        if not os.path.exists(folder2_path):
            QMessageBox.warning(self, "Warning", "After fill cleanup image folder does not exist")
            return

        # 获取nodata设置
        nodata_value = self.diff_nodata_combo.currentIndex()

        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)

        self.log_difference_message("=" * 60)
        self.log_difference_message("开始批量红色差值分析 (保留地理信息)")
        self.log_difference_message(f"种子生长前文件夹: {folder1_path}")
        self.log_difference_message(f"填充清理后文件夹: {folder2_path}")
        self.log_difference_message(f"输出文件夹: {output_folder}")
        self.log_difference_message(f"背景设置: {self.diff_nodata_combo.currentText()}")

        # 更新UI状态
        self.diff_start_btn.setEnabled(False)
        self.diff_stop_btn.setEnabled(True)
        self.diff_preview_btn.setEnabled(False)
        self.batch_processing_active = True

        # 定义进度回调函数
        def progress_callback(current, total, filename, status):
            if hasattr(self, 'batch_processing_active') and self.batch_processing_active:
                progress = int((current / total) * 100)
                self.diff_progress_bar.setValue(progress)
                self.diff_progress_bar.setFormat(f"Processing... ({current}/{total})")
                self.diff_current_status.setText(f"Status: {filename} - {status}")
                QApplication.processEvents()  # 更新UI

        try:
            # 执行批量差值分析（启用调试模式）
            success_count, failure_count, results = RedPixelDifferenceAnalyzer.batch_difference_analysis(
                folder1_path, folder2_path, output_folder, nodata_value, progress_callback, debug=True
            )

            # 生成详细统计信息
            total_files = success_count + failure_count
            total_diff_pixels = 0

            for result in results:
                if result['success'] and 'stats' in result:
                    total_diff_pixels += result['stats']['difference_pixels']

            # 记录结果
            self.log_difference_message("批量差值分析完成！")
            self.log_difference_message(f"总文件对数: {total_files}")
            self.log_difference_message(f"成功处理: {success_count}")
            self.log_difference_message(f"处理失败: {failure_count}")
            self.log_difference_message(f"总差值像素: {total_diff_pixels:,}")

            # 记录详细结果
            for result in results:
                if result['success']:
                    stats = result['stats']
                    self.log_difference_message(
                        f"✓ {result['base_name']}: 差值像素 {stats['difference_pixels']:,}, "
                        f"种子前 {stats['image1_red_pixels']:,}, 填充后 {stats['image2_red_pixels']:,}"
                    )
                else:
                    self.log_difference_message(f"✗ {result['base_name']}: {result.get('error', '未知错误')}")

            self.log_difference_message("=" * 60)

            # 更新进度条
            self.diff_progress_bar.setValue(100)
            self.diff_progress_bar.setFormat(f"完成 ({total_files}/{total_files})")
            self.diff_current_status.setText(f"Status: Batch processing complete - Success {success_count}, Failed {failure_count}")

            # Show completion dialog
            QMessageBox.information(self, "Batch Processing Complete",
                                    f"Batch red difference analysis complete! (Geographic information preserved)\n"
                                    f"Total file pairs: {total_files}\n"
                                    f"Successfully processed: {success_count}\n"
                                    f"Processing failed: {failure_count}\n"
                                    f"Total difference pixels: {total_diff_pixels:,}\n"
                                    f"Results saved to: {output_folder}")

        except Exception as e:
            self.log_difference_message(f"Error occurred during batch processing: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error occurred during batch processing:\n{str(e)}")

        finally:
            # 重置UI状态
            self.diff_start_btn.setEnabled(True)
            self.diff_stop_btn.setEnabled(False)
            self.diff_preview_btn.setEnabled(True)
            self.batch_processing_active = False

    def stop_batch_difference_analysis(self):
        """停止批量差值分析"""
        self.batch_processing_active = False
        self.diff_start_btn.setEnabled(True)
        self.diff_stop_btn.setEnabled(False)
        self.diff_preview_btn.setEnabled(True)
        self.diff_current_status.setText("Status: User stopped processing")
        self.log_difference_message("User stopped batch processing")

    def start_difference_analysis(self):
        """开始红色差值分析（保留旧方法以防兼容性问题）"""
        # 这个方法现在调用批量处理
        self.start_batch_difference_analysis()

    def select_single_file(self, line_edit, file_type):
        """选择单个文件（已弃用，保留以防兼容性问题）"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, f"选择{file_type}", "", "GeoTIFF files (*.tif *.tiff *.TIF *.TIFF)"
        )
        if file_path:
            line_edit.setText(file_path)
            self.log_difference_message(f"已选择{file_type}: {file_path}")

    def select_output_file(self):
        """选择输出文件（已弃用，保留以防兼容性问题）"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存差值分析结果", "", "GeoTIFF files (*.tif *.tiff)"
        )
        if file_path:
            self.diff_output_edit.setText(file_path)
            self.log_difference_message(f"已设置输出文件: {file_path}")

    def log_difference_message(self, message):
        """添加差值分析日志消息"""
        timestamp = time.strftime("%H:%M:%S")
        self.diff_log_text.append(f"[{timestamp}] {message}")
        self.diff_log_text.ensureCursorVisible()

    def clear_difference_log(self):
        """清除差值分析日志"""
        self.diff_log_text.clear()
        self.log_difference_message("日志已清除")

    def log_message(self, message, tool):
        """添加日志消息"""
        timestamp = time.strftime("%H:%M:%S")
        log_text = self.filler_log_text if tool == 'filler' else self.cleaner_log_text
        log_text.append(f"[{timestamp}] {message}")
        log_text.ensureCursorVisible()

    def clear_log(self, tool):
        """清除日志"""
        log_text = self.filler_log_text if tool == 'filler' else self.cleaner_log_text
        log_text.clear()
        self.log_message("日志已清除", tool)

    def start_processing(self, tool):
        """开始处理"""
        if tool == 'filler':
            input_folder = self.filler_input_folder_edit.text().strip()
            output_folder = self.filler_output_folder_edit.text().strip()
            params = {
                'mean_threshold': self.filler_mean_threshold_spin.value(),
                'std_threshold': self.filler_std_threshold_spin.value(),
                'small_gap_threshold': self.filler_small_gap_spin.value(),
                'batch_size': self.filler_batch_size_spin.value(),
                'use_green_fill': self.filler_fill_mode_combo.currentIndex() == 1
            }
            start_btn = self.filler_start_btn
            stop_btn = self.filler_stop_btn
            worker_func = worker_process_filler
            prefix = "filled_"
        else:  # cleaner
            input_folder = self.cleaner_input_folder_edit.text().strip()
            output_folder = self.cleaner_output_folder_edit.text().strip()
            color_map = {0: (255, 255, 255), 1: (0, 0, 0), 2: (128, 128, 128)}
            params = {
                'min_area_threshold': self.cleaner_min_area_spin.value(),
                'white_color': color_map[self.cleaner_replace_color_combo.currentIndex()]
            }
            start_btn = self.cleaner_start_btn
            stop_btn = self.cleaner_stop_btn
            worker_func = worker_process_cleaner
            prefix = "cleaned_"

        # Validate input
        if not input_folder or not output_folder:
            QMessageBox.warning(self, "Warning", "Please select input and output folders")
            return

        if not os.path.exists(input_folder):
            QMessageBox.warning(self, "Warning", "Input folder does not exist")
            return

        # Get TIF files
        tif_files_set = set()
        for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
            for file_path in Path(input_folder).glob(ext):
                tif_files_set.add(file_path)

        tif_files = sorted(list(tif_files_set))

        if not tif_files:
            QMessageBox.warning(self, "Warning", "No GeoTIFF files found")
            return

        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)

        # 分配文件
        file_chunks = [[] for _ in range(8)]
        for i, tif_file in enumerate(tif_files):
            process_id = i % 8
            file_chunks[process_id].append(tif_file)

        # 记录开始信息
        tool_name = "轮廓填充" if tool == 'filler' else "连通域清理"
        self.log_message("=" * 50, tool)
        self.log_message(f"开始8进程{tool_name} (保留地理信息)", tool)
        self.log_message(f"输入文件夹: {input_folder}", tool)
        self.log_message(f"输出文件夹: {output_folder}", tool)
        self.log_message(f"总GeoTIFF文件数: {len(tif_files)}", tool)
        self.log_message(f"处理参数: {params}", tool)

        # 创建队列和启动进程
        self.progress_queues[tool] = mp.Queue()
        self.processes[tool] = []

        for i in range(8):
            if file_chunks[i]:
                process = mp.Process(
                    target=worker_func,
                    args=(i, file_chunks[i], output_folder, params, self.progress_queues[tool])
                )
                process.start()
                self.processes[tool].append(process)
                self.log_message(f"启动{tool_name}进程 {i + 1} (GeoTIFF)", tool)

        # 启动监控线程
        self.monitor_threads[tool] = ProgressMonitorThread(
            self.progress_queues[tool], len(self.processes[tool]), tool
        )
        self.monitor_threads[tool].progress_updated.connect(self.update_process_progress)
        self.monitor_threads[tool].log_message.connect(self.log_message)
        self.monitor_threads[tool].all_complete.connect(self.processing_finished)
        self.monitor_threads[tool].start()

        # 更新UI状态
        start_btn.setEnabled(False)
        stop_btn.setEnabled(True)
        self.status_bar.showMessage(f"正在运行{tool_name} (保留地理信息)...")

    def stop_processing(self, tool):
        """停止处理"""
        tool_name = "轮廓填充" if tool == 'filler' else "连通域清理"
        self.log_message(f"正在停止所有{tool_name}进程...", tool)

        # 终止进程
        for process in self.processes[tool]:
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
                if process.is_alive():
                    process.kill()

        # 停止监控线程
        if self.monitor_threads[tool]:
            self.monitor_threads[tool].stop()
            self.monitor_threads[tool].wait(3000)

        # 重置UI状态
        if tool == 'filler':
            self.filler_start_btn.setEnabled(True)
            self.filler_stop_btn.setEnabled(False)
            widgets = self.filler_process_widgets
        else:
            self.cleaner_start_btn.setEnabled(True)
            self.cleaner_stop_btn.setEnabled(False)
            widgets = self.cleaner_process_widgets

        for widget in widgets:
            widget.update_progress("", "已停止", 0)

        self.processes[tool].clear()
        self.status_bar.showMessage(f"{tool_name}进程已停止")
        self.log_message(f"所有{tool_name}进程已停止", tool)

    def update_process_progress(self, process_id, filename, status, progress, total, tool):
        """更新进程进度"""
        widgets = self.filler_process_widgets if tool == 'filler' else self.cleaner_process_widgets
        if 0 <= process_id < len(widgets):
            widgets[process_id].update_progress(filename, status, progress)

    def processing_finished(self, summary, tool):
        """处理完成"""
        tool_name = "轮廓填充" if tool == 'filler' else "连通域清理"
        self.log_message(summary, tool)
        self.log_message("=" * 50, tool)

        # 重置UI状态
        if tool == 'filler':
            self.filler_start_btn.setEnabled(True)
            self.filler_stop_btn.setEnabled(False)
        else:
            self.cleaner_start_btn.setEnabled(True)
            self.cleaner_stop_btn.setEnabled(False)

        self.status_bar.showMessage(f"{tool_name}处理完成 (已保留地理信息)")
        QMessageBox.information(self, f"{tool_name}完成", summary)

        self.processes[tool].clear()

    def closeEvent(self, event):
        """窗口关闭事件"""
        running_processes = []
        for tool in ['filler', 'cleaner']:
            if self.processes[tool]:
                running_processes.append(tool)

        if running_processes:
            tools_str = " and ".join(["Contour Fill" if t == 'filler' else "Connected Component Cleanup" for t in running_processes])
            reply = QMessageBox.question(self, "Confirm Exit",
                                         f"There are still {tools_str} processes running. Force exit?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                for tool in running_processes:
                    self.stop_processing(tool)
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """主函数"""
    # 设置多进程启动方法
    mp.set_start_method('spawn', force=True)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # 设置应用程序信息
    app.setApplicationName("8进程GeoTIFF处理工具")
    app.setApplicationVersion("3.0")
    app.setOrganizationName("AI助手")

    # 创建主窗口
    window = DualToolImageProcessorGUI()
    window.show()

    # 启动应用程序
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
import sys
import os
import cv2
import numpy as np
import time
from pathlib import Path
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
                elif image.shape[-1] == 4:
                    # RGBA to BGRA
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
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


class EdgeGuidedProcessor:
    """Edge-guided pixel conversion core processing class"""

    @staticmethod
    def extract_red_pixels(image):
        """Extract red pixel mask from image"""
        try:
            # Convert to HSV for red detection
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # HSV range for red
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)

            return red_mask
        except Exception as e:
            print(f"Error extracting red pixels: {str(e)}")
            return None

    @staticmethod
    def sobel_edge_detection(red_mask):
        """Perform Sobel edge detection on red mask"""
        try:
            # Apply Gaussian filter to reduce noise
            blurred = cv2.GaussianBlur(red_mask, (3, 3), 0)

            # Sobel operator for edge detection
            sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

            # Calculate gradient magnitude
            sobel_combined = np.sqrt(sobelx ** 2 + sobely ** 2)

            # Normalize to 0-255
            sobel_normalized = np.uint8(255 * sobel_combined / np.max(sobel_combined))

            # Binarize edges
            _, edge_binary = cv2.threshold(sobel_normalized, 50, 255, cv2.THRESH_BINARY)

            return edge_binary
        except Exception as e:
            print(f"Error in Sobel edge detection: {str(e)}")
            return None

    @staticmethod
    def expand_edges(edge_mask, expansion_pixels=5):
        """Dilate edge lines, expand by specified pixel range"""
        try:
            # Create dilation kernel
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                               (expansion_pixels * 2 + 1, expansion_pixels * 2 + 1))

            # Dilation operation
            expanded_edges = cv2.dilate(edge_mask, kernel, iterations=1)

            return expanded_edges
        except Exception as e:
            print(f"Error expanding edges: {str(e)}")
            return None

    @staticmethod
    def extract_white_pixels_in_region(mask_image, region_mask):
        """Extract white pixels in specified region"""
        try:
            # Detect white pixels (assuming white pixel values close to 255)
            if len(mask_image.shape) == 3 and mask_image.shape[2] > 1:
                # If multi-channel image, convert to grayscale
                gray_mask = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
            elif len(mask_image.shape) == 3 and mask_image.shape[2] == 1:
                # If 3D but only 1 channel, squeeze dimension
                gray_mask = mask_image.squeeze(axis=2)
            else:
                # Already single-channel grayscale image
                gray_mask = mask_image

            # White pixel mask (threshold can be adjusted)
            white_mask = gray_mask > 200  # Assume white pixel values greater than 200

            # White pixels in specified region
            white_in_region = white_mask & (region_mask > 0)

            return white_in_region
        except Exception as e:
            print(f"Error extracting white pixels: {str(e)}")
            return None

    @staticmethod
    def process_single_file_pair(red_diff_path, mask_path, output_path, expansion_pixels=5, debug=False):
        """
        Process single file pair

        Args:
            red_diff_path: Red difference image path
            mask_path: Black and white mask image path
            output_path: Output path
            expansion_pixels: Edge expansion pixel count
            debug: Whether to output debug information

        Returns:
            (success status, statistics)
        """
        try:
            start_time = time.time()

            if debug:
                print(f"Processing file pair:")
                print(f"  Red difference image: {red_diff_path}")
                print(f"  Black and white mask image: {mask_path}")
                print(f"  Output path: {output_path}")

            # Read red difference image
            red_diff_image, profile = GeoTIFFHandler.read_geotiff(red_diff_path)
            if red_diff_image is None or profile is None:
                if debug:
                    print(f"  Error: Unable to read red difference image")
                return False, None

            # Read black and white mask image
            mask_image, _ = GeoTIFFHandler.read_geotiff(mask_path)
            if mask_image is None:
                if debug:
                    print(f"  Error: Unable to read black and white mask image")
                return False, None

            if debug:
                print(f"  Red difference image size: {red_diff_image.shape}")
                print(f"  Black and white mask image size: {mask_image.shape}")

            # Ensure both images have same size
            if red_diff_image.shape[:2] != mask_image.shape[:2]:
                if debug:
                    print(f"  Resizing mask image")
                mask_image = cv2.resize(mask_image, (red_diff_image.shape[1], red_diff_image.shape[0]))

            # Extract red pixels
            red_mask = EdgeGuidedProcessor.extract_red_pixels(red_diff_image)
            if red_mask is None:
                if debug:
                    print(f"  Error: Unable to extract red pixels")
                return False, None

            red_pixel_count = np.sum(red_mask > 0)
            if debug:
                print(f"  Red pixel count: {red_pixel_count:,}")

            if red_pixel_count == 0:
                if debug:
                    print(f"  Warning: No red pixels, copying original image")
                # No red pixels, directly copy original image
                success = GeoTIFFHandler.write_geotiff(output_path, red_diff_image, profile)
                stats = {
                    'red_pixels': 0,
                    'edge_pixels': 0,
                    'converted_pixels': 0,
                    'processing_time': time.time() - start_time
                }
                return success, stats

            # 对红色区域进行Sobel边缘检测
            edge_mask = EdgeGuidedProcessor.sobel_edge_detection(red_mask)
            if edge_mask is None:
                if debug:
                    print(f"  错误: Sobel边缘检测失败")
                return False, None

            edge_pixel_count = np.sum(edge_mask > 0)
            if debug:
                print(f"  边缘像素数量: {edge_pixel_count:,}")

            # 扩展边缘
            expanded_edges = EdgeGuidedProcessor.expand_edges(edge_mask, expansion_pixels)
            if expanded_edges is None:
                if debug:
                    print(f"  错误: 边缘扩展失败")
                return False, None

            expanded_pixel_count = np.sum(expanded_edges > 0)
            if debug:
                print(f"  扩展后边缘像素数量: {expanded_pixel_count:,}")

            # 在扩展边缘区域内查找白色像素
            white_pixels_in_region = EdgeGuidedProcessor.extract_white_pixels_in_region(
                mask_image, expanded_edges
            )
            if white_pixels_in_region is None:
                if debug:
                    print(f"  错误: 提取白色像素失败")
                return False, None

            converted_pixel_count = np.sum(white_pixels_in_region)
            if debug:
                print(f"  找到的白色像素数量: {converted_pixel_count:,}")

            # 创建输出图像（复制原始红色差值图像）
            result_image = red_diff_image.copy()

            # 将找到的白色像素转换为红色
            if converted_pixel_count > 0:
                # 根据图像通道数设置正确的红色值
                if result_image.shape[2] == 4:
                    # 4通道图像（BGRA）
                    red_color = [0, 0, 255, 255]  # BGRA格式的红色（不透明）
                elif result_image.shape[2] == 3:
                    # 3通道图像（BGR）
                    red_color = [0, 0, 255]  # BGR格式的红色
                else:
                    # 处理其他情况
                    red_color = [255]  # 单通道时使用白色

                result_image[white_pixels_in_region] = red_color
                if debug:
                    print(f"  已将 {converted_pixel_count:,} 个白色像素转换为红色")
                    print(f"  使用颜色值: {red_color} (通道数: {result_image.shape[2]})")

            # 保存结果图像
            success = GeoTIFFHandler.write_geotiff(output_path, result_image, profile)

            processing_time = time.time() - start_time

            if debug:
                print(f"  保存成功: {success}")
                print(f"  处理时间: {processing_time:.2f}秒")

            stats = {
                'red_pixels': red_pixel_count,
                'edge_pixels': edge_pixel_count,
                'converted_pixels': converted_pixel_count,
                'processing_time': processing_time
            }

            return success, stats

        except Exception as e:
            if debug:
                print(f"  处理出错: {str(e)}")
            print(f"处理文件对时出错: {str(e)}")
            return False, None

    @staticmethod
    def batch_process(red_diff_folder, mask_base_folder, output_folder, expansion_pixels=5,
                      progress_callback=None, debug=False):
        """
        批量处理边缘引导像素转换

        Args:
            red_diff_folder: 红色差值图像文件夹
            mask_base_folder: 黑白掩膜基础文件夹
            output_folder: 输出文件夹
            expansion_pixels: 边缘扩展像素数
            progress_callback: 进度回调函数
            debug: 是否启用调试模式

        Returns:
            (成功数量, 失败数量, 详细结果列表)
        """
        try:
            if debug:
                print(f"开始批量边缘引导处理:")
                print(f"  红色差值文件夹: {red_diff_folder}")
                print(f"  黑白掩膜基础文件夹: {mask_base_folder}")
                print(f"  输出文件夹: {output_folder}")
                print(f"  边缘扩展像素: {expansion_pixels}")

            # 获取红色差值文件夹中的所有tif文件
            red_diff_files = {}
            for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
                for file_path in Path(red_diff_folder).glob(ext):
                    filename = file_path.stem
                    if filename.startswith('diff_'):
                        # 提取XXX_XXX模式
                        base_name = filename.replace('diff_', '')
                        red_diff_files[base_name] = file_path

            if debug:
                print(f"  找到 {len(red_diff_files)} 个红色差值文件:")
                for base_name in list(red_diff_files.keys())[:5]:
                    print(f"    {base_name}")
                if len(red_diff_files) > 5:
                    print(f"    ... 还有 {len(red_diff_files) - 5} 个文件")

            # 查找匹配的掩膜文件
            matched_pairs = []
            for base_name, red_diff_path in red_diff_files.items():
                # 构建掩膜文件路径
                mask_folder = os.path.join(mask_base_folder, f"tile_{base_name}_normalized_processed")
                mask_path = os.path.join(mask_folder, "step7_filtered_mask3.tif")

                if os.path.exists(mask_path):
                    matched_pairs.append((base_name, red_diff_path, mask_path))
                    if debug and len(matched_pairs) <= 3:
                        print(f"    匹配: {base_name} -> {mask_path}")

            if debug:
                print(f"  找到 {len(matched_pairs)} 个匹配的文件对")

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

            for i, (base_name, red_diff_path, mask_path) in enumerate(matched_pairs):
                try:
                    # 构建输出文件路径
                    output_filename = f"edge_guided_{base_name}.tif"
                    output_path = os.path.join(output_folder, output_filename)

                    # 调用进度回调
                    if progress_callback:
                        progress_callback(i + 1, total_pairs, base_name, "处理中...")

                    if debug:
                        print(f"\n处理第 {i + 1}/{total_pairs} 个文件对: {base_name}")

                    # 执行边缘引导处理
                    success, stats = EdgeGuidedProcessor.process_single_file_pair(
                        str(red_diff_path), mask_path, output_path, expansion_pixels, debug=debug
                    )

                    if success and stats:
                        success_count += 1
                        result_info = {
                            'base_name': base_name,
                            'red_diff_file': str(red_diff_path),
                            'mask_file': mask_path,
                            'output': output_path,
                            'success': True,
                            'stats': stats
                        }
                        results.append(result_info)

                        # 更新进度
                        if progress_callback:
                            progress_callback(i + 1, total_pairs, base_name,
                                              f"成功 - 转换 {stats['converted_pixels']:,} 像素")
                    else:
                        failure_count += 1
                        result_info = {
                            'base_name': base_name,
                            'red_diff_file': str(red_diff_path),
                            'mask_file': mask_path,
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
                        'red_diff_file': str(red_diff_path),
                        'mask_file': mask_path,
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
                print(f"批量边缘引导处理时出错: {str(e)}")
            print(f"批量边缘引导处理时出错: {str(e)}")
            return 0, 0, []


class EdgeGuidedProcessorGUI(QMainWindow):
    """边缘引导像素转换工具用户界面"""

    def __init__(self):
        super().__init__()
        self.batch_processing_active = False
        self.init_ui()

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("Edge-Guided Pixel Converter Tool")
        self.setGeometry(100, 100, 1200, 800)

        # 设置样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
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
        main_layout = QHBoxLayout(central_widget)

        # 左侧控制面板
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)

        # 右侧日志面板
        right_panel = self.create_log_panel()
        main_layout.addWidget(right_panel, 1)

        # 底部状态栏
        self.create_status_bar()

    def create_status_bar(self):
        """创建状态栏"""
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Edge-Guided Pixel Converter Tool Started - Using Sobel Edge Detection")

    def create_control_panel(self):
        """创建控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Folder selection group
        folder_group = QGroupBox("Folder Settings")
        folder_layout = QGridLayout(folder_group)

        # Red difference folder
        folder_layout.addWidget(QLabel("Red Difference Folder:"), 0, 0)
        self.red_diff_folder_edit = QLineEdit()
        self.red_diff_folder_edit.setPlaceholderText("Select folder containing diff_XXX_XXX.tif files")
        folder_layout.addWidget(self.red_diff_folder_edit, 1, 0, 1, 2)
        red_diff_btn = QPushButton("Browse")
        red_diff_btn.setStyleSheet("QPushButton { background-color: #E91E63; }")
        red_diff_btn.clicked.connect(lambda: self.select_folder(self.red_diff_folder_edit, "Red Difference"))
        folder_layout.addWidget(red_diff_btn, 1, 2)

        # Black and white mask base folder
        folder_layout.addWidget(QLabel("Black & White Mask Base Folder:"), 2, 0)
        self.mask_base_folder_edit = QLineEdit()
        self.mask_base_folder_edit.setPlaceholderText("Select base folder containing tile_XXX_XXX_normalized_processed subfolders")
        folder_layout.addWidget(self.mask_base_folder_edit, 3, 0, 1, 2)
        mask_base_btn = QPushButton("Browse")
        mask_base_btn.setStyleSheet("QPushButton { background-color: #E91E63; }")
        mask_base_btn.clicked.connect(lambda: self.select_folder(self.mask_base_folder_edit, "Mask Base"))
        folder_layout.addWidget(mask_base_btn, 3, 2)

        # Output folder
        folder_layout.addWidget(QLabel("Output Folder:"), 4, 0)
        self.output_folder_edit = QLineEdit()
        self.output_folder_edit.setPlaceholderText("Select output folder")
        folder_layout.addWidget(self.output_folder_edit, 5, 0, 1, 2)
        output_btn = QPushButton("Browse")
        output_btn.setStyleSheet("QPushButton { background-color: #E91E63; }")
        output_btn.clicked.connect(lambda: self.select_folder(self.output_folder_edit, "Output"))
        folder_layout.addWidget(output_btn, 5, 2)

        layout.addWidget(folder_group)

        # Processing parameters settings group
        params_group = QGroupBox("Processing Parameters")
        params_layout = QGridLayout(params_group)

        # Edge expansion pixels
        params_layout.addWidget(QLabel("Edge Expansion Pixels:"), 0, 0)
        self.expansion_pixels_spin = QSpinBox()
        self.expansion_pixels_spin.setRange(1, 20)
        self.expansion_pixels_spin.setValue(5)
        self.expansion_pixels_spin.setSuffix(" pixels")
        self.expansion_pixels_spin.setToolTip("Pixel range for Sobel edge outward expansion")
        params_layout.addWidget(self.expansion_pixels_spin, 0, 1)

        layout.addWidget(params_group)

        # Batch processing progress group
        progress_group = QGroupBox("Batch Processing Progress")
        progress_layout = QVBoxLayout(progress_group)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("Waiting to start... (0/0)")
        progress_layout.addWidget(self.progress_bar)

        # Current processing status
        self.current_status = QLabel("Status: Waiting to start")
        self.current_status.setStyleSheet("color: #666; font-size: 10px;")
        progress_layout.addWidget(self.current_status)

        layout.addWidget(progress_group)

        # Processing flow description group
        flow_group = QGroupBox("Processing Flow Description")
        flow_layout = QVBoxLayout(flow_group)

        flow_label = QLabel("""
<b>Edge-Guided Pixel Conversion Process:</b><br>
<font color="#E91E63">1. Read red difference image (diff_XXX_XXX.tif)</font><br>
<font color="#673AB7">2. Extract red pixels and perform Sobel edge detection</font><br>
<font color="#3F51B5">3. Expand edges by specified pixel range</font><br>
<font color="#009688">4. Match black & white mask image (step7_filtered_mask3.tif)</font><br>
<font color="#4CAF50">5. Find white pixels within edge range</font><br>
<font color="#FF5722">6. Convert white pixels to red pixels</font><br><br>

<b>File Matching Rules:</b><br>
• diff_000_002.tif → tile_000_002_normalized_processed/step7_filtered_mask3.tif<br>
• diff_000_003.tif → tile_000_003_normalized_processed/step7_filtered_mask3.tif<br><br>

<b>✓ Use rasterio to preserve geographic information</b><br>
<b>✓ Sobel operator for high-precision edge detection</b><br>
<b>✓ Morphological dilation for precise expansion control</b>
        """)
        flow_label.setStyleSheet("color: #444; font-size: 10px;")
        flow_layout.addWidget(flow_label)

        layout.addWidget(flow_group)

        # Control buttons group
        control_group = QGroupBox("Operation Control")
        control_layout = QVBoxLayout(control_group)

        # Preview file matches button
        self.preview_btn = QPushButton("Preview File Match Results")
        self.preview_btn.setStyleSheet("QPushButton { background-color: #607D8B; }")
        self.preview_btn.clicked.connect(self.preview_file_matches)
        control_layout.addWidget(self.preview_btn)

        # Start batch processing button
        self.start_btn = QPushButton("Start Batch Edge-Guided Processing")
        self.start_btn.setStyleSheet("QPushButton { background-color: #E91E63; }")
        self.start_btn.clicked.connect(self.start_batch_processing)
        control_layout.addWidget(self.start_btn)

        # Stop processing button
        self.stop_btn = QPushButton("Stop Batch Processing")
        self.stop_btn.setStyleSheet("QPushButton { background-color: #f44336; }")
        self.stop_btn.clicked.connect(self.stop_batch_processing)
        self.stop_btn.setEnabled(False)
        control_layout.addWidget(self.stop_btn)

        layout.addWidget(control_group)
        layout.addStretch()

        return panel

    def create_log_panel(self):
        """创建日志面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        log_label = QLabel("Processing Log")
        log_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(log_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.log_text)

        clear_btn = QPushButton("Clear Log")
        clear_btn.setStyleSheet("QPushButton { background-color: #ff9800; }")
        clear_btn.clicked.connect(self.clear_log)
        layout.addWidget(clear_btn)

        return panel

    def select_folder(self, line_edit, folder_type):
        """Select folder"""
        folder = QFileDialog.getExistingDirectory(self, f"Select {folder_type} Folder")
        if folder:
            line_edit.setText(folder)
            self.log_message(f"Selected {folder_type} folder: {folder}")

    def preview_file_matches(self):
        """预览文件匹配结果"""
        red_diff_folder = self.red_diff_folder_edit.text().strip()
        mask_base_folder = self.mask_base_folder_edit.text().strip()

        if not red_diff_folder or not mask_base_folder:
            QMessageBox.warning(self, "Warning", "Please select red difference folder and black & white mask base folder first")
            return

        if not os.path.exists(red_diff_folder) or not os.path.exists(mask_base_folder):
            QMessageBox.warning(self, "Warning", "Selected folder does not exist")
            return

        try:
            # 获取红色差值文件夹中的所有tif文件
            red_diff_files = {}
            for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
                for file_path in Path(red_diff_folder).glob(ext):
                    filename = file_path.stem
                    if filename.startswith('diff_'):
                        base_name = filename.replace('diff_', '')
                        red_diff_files[base_name] = file_path

            # 查找匹配的掩膜文件
            matched_pairs = []
            unmatched_files = []

            for base_name, red_diff_path in red_diff_files.items():
                mask_folder = os.path.join(mask_base_folder, f"tile_{base_name}_normalized_processed")
                mask_path = os.path.join(mask_folder, "step7_filtered_mask3.tif")

                if os.path.exists(mask_path):
                    matched_pairs.append((base_name, red_diff_path.name, mask_path))
                else:
                    unmatched_files.append((base_name, red_diff_path.name, mask_path))

            # 生成预览信息
            preview_text = f"""文件匹配预览结果：

【匹配的文件对】({len(matched_pairs)} 对)：
"""
            for base_name, red_file, mask_path in matched_pairs:
                preview_text += f"• {base_name}:\n"
                preview_text += f"  红色差值: {red_file}\n"
                preview_text += f"  黑白掩膜: {mask_path}\n\n"

            if unmatched_files:
                preview_text += f"【无匹配掩膜的文件】({len(unmatched_files)} 个)：\n"
                for base_name, red_file, expected_mask_path in unmatched_files:
                    preview_text += f"• {base_name}: {red_file}\n"
                    preview_text += f"  期望掩膜路径: {expected_mask_path}\n\n"

            preview_text += f"将处理 {len(matched_pairs)} 个文件对进行边缘引导像素转换。"

            # Show preview dialog
            msg = QMessageBox()
            msg.setWindowTitle("File Match Preview")
            msg.setText("File Match Preview Results")
            msg.setDetailedText(preview_text)
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()

            # Log to log
            self.log_message(f"Preview complete: Found {len(matched_pairs)} matched file pairs")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error previewing file matches:\n{str(e)}")
            self.log_message(f"Error previewing file matches: {str(e)}")

    def start_batch_processing(self):
        """开始批量边缘引导处理"""
        red_diff_folder = self.red_diff_folder_edit.text().strip()
        mask_base_folder = self.mask_base_folder_edit.text().strip()
        output_folder = self.output_folder_edit.text().strip()

        # Validate input
        if not red_diff_folder or not mask_base_folder or not output_folder:
            QMessageBox.warning(self, "Warning", "Please select all required folders")
            return

        if not os.path.exists(red_diff_folder):
            QMessageBox.warning(self, "Warning", "Red difference folder does not exist")
            return

        if not os.path.exists(mask_base_folder):
            QMessageBox.warning(self, "Warning", "Black & white mask base folder does not exist")
            return

        # 获取参数
        expansion_pixels = self.expansion_pixels_spin.value()

        # 创建输出文件夹
        os.makedirs(output_folder, exist_ok=True)

        self.log_message("=" * 60)
        self.log_message("开始批量边缘引导像素转换 (保留地理信息)")
        self.log_message(f"红色差值文件夹: {red_diff_folder}")
        self.log_message(f"黑白掩膜基础文件夹: {mask_base_folder}")
        self.log_message(f"输出文件夹: {output_folder}")
        self.log_message(f"边缘扩展像素: {expansion_pixels}")

        # 更新UI状态
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.preview_btn.setEnabled(False)
        self.batch_processing_active = True

        # 定义进度回调函数
        def progress_callback(current, total, filename, status):
            if hasattr(self, 'batch_processing_active') and self.batch_processing_active:
                progress = int((current / total) * 100)
                self.progress_bar.setValue(progress)
                self.progress_bar.setFormat(f"Processing... ({current}/{total})")
                self.current_status.setText(f"Status: {filename} - {status}")
                QApplication.processEvents()  # 更新UI

        try:
            # 执行批量边缘引导处理
            success_count, failure_count, results = EdgeGuidedProcessor.batch_process(
                red_diff_folder, mask_base_folder, output_folder, expansion_pixels,
                progress_callback, debug=True
            )

            # 生成详细统计信息
            total_files = success_count + failure_count
            total_converted_pixels = 0
            total_red_pixels = 0
            total_edge_pixels = 0

            for result in results:
                if result['success'] and 'stats' in result:
                    stats = result['stats']
                    total_converted_pixels += stats['converted_pixels']
                    total_red_pixels += stats['red_pixels']
                    total_edge_pixels += stats['edge_pixels']

            # 记录结果
            self.log_message("批量边缘引导处理完成！")
            self.log_message(f"总文件对数: {total_files}")
            self.log_message(f"成功处理: {success_count}")
            self.log_message(f"处理失败: {failure_count}")
            self.log_message(f"总红色像素: {total_red_pixels:,}")
            self.log_message(f"总边缘像素: {total_edge_pixels:,}")
            self.log_message(f"总转换像素: {total_converted_pixels:,}")

            # 记录详细结果
            for result in results:
                if result['success']:
                    stats = result['stats']
                    self.log_message(
                        f"✓ {result['base_name']}: 转换 {stats['converted_pixels']:,} 像素, "
                        f"红色 {stats['red_pixels']:,}, 边缘 {stats['edge_pixels']:,}"
                    )
                else:
                    self.log_message(f"✗ {result['base_name']}: {result.get('error', '未知错误')}")

            self.log_message("=" * 60)

            # Update progress bar
            self.progress_bar.setValue(100)
            self.progress_bar.setFormat(f"Complete ({total_files}/{total_files})")
            self.current_status.setText(f"Status: Batch processing complete - Success {success_count}, Failed {failure_count}")

            # Show completion dialog
            QMessageBox.information(self, "Batch Processing Complete",
                                    f"Batch edge-guided processing complete! (Geographic information preserved)\n"
                                    f"Total file pairs: {total_files}\n"
                                    f"Successfully processed: {success_count}\n"
                                    f"Processing failed: {failure_count}\n"
                                    f"Total converted pixels: {total_converted_pixels:,}\n"
                                    f"Results saved to: {output_folder}")

        except Exception as e:
            self.log_message(f"Error occurred during batch processing: {str(e)}")
            QMessageBox.critical(self, "Error", f"Error occurred during batch processing:\n{str(e)}")

        finally:
            # 重置UI状态
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.preview_btn.setEnabled(True)
            self.batch_processing_active = False

    def stop_batch_processing(self):
        """停止批量处理"""
        self.batch_processing_active = False
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.preview_btn.setEnabled(True)
        self.current_status.setText("Status: User stopped processing")
        self.log_message("User stopped batch processing")

    def log_message(self, message):
        """添加日志消息"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {message}")
        self.log_text.ensureCursorVisible()

    def clear_log(self):
        """清除日志"""
        self.log_text.clear()
        self.log_message("Log cleared")

    def closeEvent(self, event):
        """窗口关闭事件"""
        if self.batch_processing_active:
            reply = QMessageBox.question(self, "Confirm Exit",
                                         "Batch processing is in progress. Force exit?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.batch_processing_active = False
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


def main():
    """主函数"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # 设置应用程序信息
    app.setApplicationName("Edge-Guided Pixel Converter Tool")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("AI Assistant")

    # 创建主窗口
    window = EdgeGuidedProcessorGUI()
    window.show()

    # 启动应用程序
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
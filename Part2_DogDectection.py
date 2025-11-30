import sys
import os
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import threading
import time
import multiprocessing as mp
from multiprocessing import Manager  # Additional import for shared state
import gc
import csv

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QLineEdit, QPushButton, QFileDialog,
    QProgressBar, QTextEdit, QGroupBox, QFormLayout, QSpinBox,
    QCheckBox, QMessageBox, QTableWidget, QTableWidgetItem,
    QSplitter, QHeaderView
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QSettings, pyqtSlot
)

import numpy as np
from PIL import Image
import cv2
from scipy import ndimage
import gc
import csv
import rasterio

try:
    import pandas as pd

    try:
        import openpyxl

        OPENPYXL_AVAILABLE = True
    except ImportError:
        OPENPYXL_AVAILABLE = False
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    OPENPYXL_AVAILABLE = False

try:
    from osgeo import gdal

    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===================== Enhanced endpoint patterns (ported from Step4) =====================
PATTERNS = np.array([
    [[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    [[1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    [[0, 0, 0, 1, 0], [0, 0, 1, 1, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    [[0, 0, 0, 1, 1], [0, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    [[0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    [[0, 0, 0, 1, 1], [0, 0, 1, 1, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    [[0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    [[1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    [[0, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    [[0, 0, 0, 0, 0], [0, 0, 1, 1, 0], [0, 1, 1, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    [[0, 0, 0, 0, 1], [0, 0, 1, 1, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    [[0, 0, 0, 1, 1], [0, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    [[1, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    [[0, 0, 1, 1, 1], [0, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
], dtype=np.float32)


def generate_all_pattern_transforms(patterns):
    """Generate every transformed version of the endpoint templates (rotation + flip)"""
    all_transforms = []

    for pattern in patterns:
        transforms_set = set()

        # Original pattern plus its horizontal flip
        base_patterns = [pattern, np.fliplr(pattern)]

        # Apply four rotations to each base pattern
        for base_pattern in base_patterns:
            for rotation in range(4):  # 0°, 90°, 180°, 270°
                transformed = np.rot90(base_pattern, rotation)
                # Convert to tuple for deduplication
                pattern_tuple = tuple(transformed.flatten())
                transforms_set.add(pattern_tuple)

        # Convert back to array form
        for pattern_tuple in transforms_set:
            all_transforms.append(np.array(pattern_tuple).reshape(5, 5))

    return np.array(all_transforms)


def find_endpoints_vectorized(binary_image):
    """Vectorized endpoint detection (ported from Step4)"""
    # Convert to a 0/1 binary image
    binary = (binary_image == 255).astype(np.float32)
    h, w = binary.shape

    # Conventional endpoint detection (8-neighborhood count equals 1)
    kernel_8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
    neighbor_count = cv2.filter2D(binary, -1, kernel_8)
    traditional_endpoints = (binary == 1) & (neighbor_count == 1)

    # Generate all transformed patterns
    all_pattern_transforms = generate_all_pattern_transforms(PATTERNS)

    # Pattern-matching endpoints
    pattern_endpoints = np.zeros((h, w), dtype=bool)

    # Convolution-based matching for every transform
    for transformed_pattern in all_pattern_transforms:
        # Use convolution to identify matches
        pattern_mask = (transformed_pattern == 1)
        pattern_sum = np.sum(pattern_mask)

        if pattern_sum > 0:
            # Convolution check
            conv_result = cv2.filter2D(binary, -1, pattern_mask.astype(np.float32))

            # Match locations where the sum aligns and the center pixel equals 1
            matches = (conv_result == pattern_sum) & (binary == 1)

            # Exclude borders (5x5 template requires two-pixel margin)
            matches[:2, :] = False
            matches[-2:, :] = False
            matches[:, :2] = False
            matches[:, -2:] = False

            pattern_endpoints |= matches

    # Extract endpoint coordinates
    traditional_coords = np.column_stack(np.where(traditional_endpoints))
    pattern_coords = np.column_stack(np.where(pattern_endpoints & ~traditional_endpoints))

    # Combine both sets
    if len(traditional_coords) == 0 and len(pattern_coords) == 0:
        return []
    elif len(traditional_coords) == 0:
        return pattern_coords.tolist()
    elif len(pattern_coords) == 0:
        return traditional_coords.tolist()
    else:
        return np.vstack([traditional_coords, pattern_coords]).tolist()


def get_8_neighbor_points(binary_image, point):
    """
    Return all 8-neighborhood pixels whose value equals 255.
    Returns a tuple of (count, neighbor list).
    """
    y, x = point
    height, width = binary_image.shape

    # Skip if current pixel is background
    if binary_image[y, x] == 0:
        return 0, []

    neighbor_points = []

    # 8-neighborhood offsets (dy, dx)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    for dy, dx in offsets:
        ny, nx = y + dy, x + dx

        # Bounds check
        if 0 <= ny < height and 0 <= nx < width:
            if binary_image[ny, nx] == 255:
                neighbor_points.append((ny, nx))

    return len(neighbor_points), neighbor_points


class LakeDetectionConfig:
    def __init__(self):
        self.EDGE_THRESHOLD = 50
        self.THRESHOLD1 = 200
        self.KERNEL1_SIZE = 4
        self.THRESHOLD2 = 200
        self.KERNEL2_SIZE = 2
        self.THRESHOLD3 = 500
        self.ENABLE_DILATION = False
        self.DILATION_KERNEL_SIZE = 2
        self.BURR_THRESHOLD = 30
        self.SMALL_CONTOUR_THRESHOLD = 60
        self.PADDING = 5

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items()}

    def update_from_dict(self, params: Dict):
        for key, value in params.items():
            if hasattr(self, key.upper()):
                setattr(self, key.upper(), value)


class DEMProcessor:
    def __init__(self, dem_path: str, sar_path: str):
        self.dem_data = None
        self.dem_transform = None
        self.sar_transform = None
        self.load_data(dem_path, sar_path)

    def load_data(self, dem_path: str, sar_path: str):
        try:
            with rasterio.open(dem_path) as dem_src:
                self.dem_data = dem_src.read(1)
                self.dem_transform = dem_src.transform
                self.dem_nodata = dem_src.nodata

            with rasterio.open(sar_path) as sar_src:
                self.sar_transform = sar_src.transform

        except Exception as e:
            logger.error(f"Failed to load DEM/SAR data: {e}")

    def calculate_dem_statistics_for_region(self, region_mask: np.ndarray) -> Dict:
        try:
            rows, cols = np.where(region_mask)
            if len(rows) == 0:
                return {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'count': 0, 'min': 0.0, 'max': 0.0}

            # Simple coordinate transform to avoid vectorization bugs
            dem_values = []
            for sar_row, sar_col in zip(rows, cols):
                try:
                    # SAR pixel -> geographic coordinates
                    geo_x = self.sar_transform[2] + sar_col * self.sar_transform[0]
                    geo_y = self.sar_transform[5] + sar_row * self.sar_transform[4]

                    # Geographic coordinates -> DEM pixels
                    dem_col = int((geo_x - self.dem_transform[2]) / self.dem_transform[0])
                    dem_row = int((geo_y - self.dem_transform[5]) / self.dem_transform[4])

                    # Bounds check
                    if (0 <= dem_row < self.dem_data.shape[0] and
                            0 <= dem_col < self.dem_data.shape[1]):
                        dem_value = self.dem_data[dem_row, dem_col]
                        if self.dem_nodata is None or dem_value != self.dem_nodata:
                            if dem_value != 0:  # Drop zero values
                                dem_values.append(dem_value)
                except:
                    continue

            if dem_values:
                dem_values = np.array(dem_values)
                return {
                    'mean': round(float(np.mean(dem_values)), 2),
                    'std': round(float(np.std(dem_values)), 2),
                    'median': round(float(np.median(dem_values)), 2),
                    'min': round(float(np.min(dem_values)), 2),
                    'max': round(float(np.max(dem_values)), 2),
                    'count': len(dem_values)
                }
            else:
                return {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}

        except Exception as e:
            logger.warning(f"DEM statistics failed: {e}")
            return {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}


class LakeDetectionProcessor:
    def __init__(self, config: LakeDetectionConfig):
        self.config = config
        self.geotransform = None
        self.projection = None

    def read_geotiff_info(self, image_path: str):
        if GDAL_AVAILABLE:
            try:
                dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
                if dataset:
                    self.geotransform = dataset.GetGeoTransform()
                    self.projection = dataset.GetProjection()
                    dataset = None
            except Exception as e:
                logger.warning(f"Failed to read GeoTIFF metadata: {e}")

    def save_array_as_geotiff(self, array: np.ndarray, output_path: str):
        try:
            if GDAL_AVAILABLE and self.geotransform:
                height, width = array.shape
                driver = gdal.GetDriverByName('GTiff')
                dataset = driver.Create(output_path, width, height, 1, gdal.GDT_Byte)
                dataset.SetGeoTransform(self.geotransform)
                dataset.SetProjection(self.projection)
                dataset.GetRasterBand(1).WriteArray(array)
                dataset = None
            else:
                Image.fromarray(array).save(output_path)
        except Exception as e:
            logger.warning(f"GeoTIFF save failed, falling back to PIL: {e}")
            Image.fromarray(array).save(output_path)

    def apply_dog_edge_detection(self, img_array: np.ndarray) -> np.ndarray:
        g1 = ndimage.gaussian_filter(img_array, sigma=2.0)
        g2 = ndimage.gaussian_filter(img_array, sigma=6.0)
        result = g1 - g2
        result = np.absolute(result)
        result = result / result.max() * 255 if result.max() > 0 else result
        return result.astype(np.uint8)

    def filter_small_regions(self, binary_image: np.ndarray, min_area: int) -> np.ndarray:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_image, connectivity=8)
        if num_labels <= 1:
            return np.zeros_like(binary_image)

        areas = stats[1:, cv2.CC_STAT_AREA]
        valid_mask = areas >= min_area
        if not np.any(valid_mask):
            return np.zeros_like(binary_image)

        valid_labels = np.where(valid_mask)[0] + 1
        filtered_mask = np.isin(labels, valid_labels).astype(np.uint8) * 255
        return filtered_mask

    def apply_enhanced_thinning(self, binary_image: np.ndarray) -> np.ndarray:
        """Enhanced thinning algorithm (ported from Step4)."""
        try:
            # Normalize to binary mask
            binary = binary_image.copy()
            if np.max(binary) > 1:
                binary = (binary > 0).astype(np.uint8) * 255

            # Require opencv-contrib-python for ximgproc
            if hasattr(cv2, 'ximgproc') and hasattr(cv2.ximgproc, 'thinning'):
                # Apply Zhang-Suen thinning
                thinned = cv2.ximgproc.thinning(binary)

                # Optionally dilate to repair broken strokes
                if self.config.ENABLE_DILATION:
                    kernel = np.ones((self.config.DILATION_KERNEL_SIZE, self.config.DILATION_KERNEL_SIZE), np.uint8)
                    thinned = cv2.dilate(thinned, kernel, iterations=1)
                    logger.info(f'   Thinning + dilation (kernel:{self.config.DILATION_KERNEL_SIZE})')
                else:
                    logger.info('   Thinning (no dilation)')

                return thinned
            else:
                logger.warning("OpenCV contrib package unavailable, skipping thinning step")
                return binary

        except Exception as e:
            logger.warning(f"Thinning algorithm failed: {e}, returning original image")
            return binary_image

    def remove_enhanced_burr(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Enhanced burr removal using vectorized endpoint detection (ported from Step4)
        """
        # Work on a copy to avoid mutating the input
        result_image = binary_image.copy()

        # Collect every endpoint with the vectorized helper
        endpoints = find_endpoints_vectorized(result_image)

        if not endpoints:
            logger.info("   No endpoints detected")
            return result_image

        logger.info(f"   Total endpoints: {len(endpoints)}")

        removed_burr_count = 0

        for terminal_point in endpoints:
            # Skip endpoints that were already cleared
            if result_image[terminal_point[0], terminal_point[1]] == 0:
                continue

            last_pt = terminal_point
            next_pt = terminal_point
            branch_length = 0
            flag_end = True
            burr_points = []

            while flag_end:
                neighbor_count, neighbor_points = get_8_neighbor_points(result_image, next_pt)

                if neighbor_count == 1:
                    # Endpoint, keep extending
                    burr_points.append(next_pt)
                    last_pt = next_pt
                    next_pt = neighbor_points[0]
                    branch_length += 1

                elif neighbor_count == 2:
                    # Intermediate point, pick the branch that is not the previous pixel
                    burr_points.append(next_pt)
                    if last_pt != neighbor_points[0]:
                        last_pt = next_pt
                        next_pt = neighbor_points[0]
                    else:
                        last_pt = next_pt
                        next_pt = neighbor_points[1]
                    branch_length += 1

                elif neighbor_count >= 3:
                    # Junction, stop extending
                    flag_end = False

                else:
                    # neighbor_count == 0, isolated pixel
                    flag_end = False

                # Guard against infinite loops
                if branch_length > self.config.BURR_THRESHOLD:
                    flag_end = False

            # Remove short burr branches
            if branch_length < self.config.BURR_THRESHOLD and len(burr_points) > 0:
                for burr_point in burr_points:
                    result_image[burr_point[0], burr_point[1]] = 0
                removed_burr_count += 1

        logger.info(f"   Removed {removed_burr_count} burr branches in total")
        return result_image

    def calculate_contour_features(self, contour_points: np.ndarray) -> Tuple[float, float]:
        try:
            if len(contour_points) < 5:
                return 0.0, 1.0

            hull = cv2.convexHull(contour_points)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(contour_points)

            if hull_area > 1e-6:
                convexity_defect = 1 - (contour_area / hull_area)
            else:
                convexity_defect = 0

            perimeter = cv2.arcLength(contour_points, True)
            hull_perimeter = cv2.arcLength(hull, True)

            if contour_area > 1e-6:
                shape_complexity = (perimeter * perimeter) / (12.566370614359172 * contour_area)
            else:
                shape_complexity = 0

            if hull_perimeter > 1e-6:
                roughness = perimeter / hull_perimeter
            else:
                roughness = 1.0

            curvature_feature = convexity_defect * shape_complexity
            return round(curvature_feature, 4), round(roughness, 4)

        except:
            return 0.0, 1.0

    def calculate_pixel_statistics(self, region_mask: np.ndarray, original_array: np.ndarray) -> Dict:
        try:
            region_pixels = original_array[region_mask]
            if region_pixels.size == 0:
                return {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'min': 0, 'max': 0}

            return {
                'mean': round(float(np.mean(region_pixels)), 2),
                'std': round(float(np.std(region_pixels)), 2),
                'median': round(float(np.median(region_pixels)), 2),
                'min': int(np.min(region_pixels)),
                'max': int(np.max(region_pixels))
            }
        except:
            return {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'min': 0, 'max': 0}

    def process_image(self, img_array: np.ndarray, progress_callback: Optional[Callable] = None) -> Dict:
        results = {}

        # Difference of Gaussian edge detection
        if progress_callback: progress_callback(0.1, "Running DoG edge detection")
        edges = self.apply_dog_edge_detection(img_array)
        results['edges'] = edges

        # Binarization
        if progress_callback: progress_callback(0.2, "Binarizing edge map")
        binary_edges = (edges > self.config.EDGE_THRESHOLD).astype(np.uint8) * 255
        results['binary_edges'] = binary_edges

        # Connected-component filter pass 1
        if progress_callback: progress_callback(0.3, "Connected-component filter pass 1")
        filtered_mask1 = self.filter_small_regions(binary_edges, self.config.THRESHOLD1)
        results['filtered_mask1'] = filtered_mask1

        # Opening
        if progress_callback: progress_callback(0.4, "Applying morphological opening")
        kernel = np.ones((self.config.KERNEL1_SIZE, self.config.KERNEL1_SIZE), np.uint8)
        opened_mask = cv2.morphologyEx(filtered_mask1, cv2.MORPH_OPEN, kernel)
        results['opened_mask'] = opened_mask

        # Connected-component filter pass 2
        if progress_callback: progress_callback(0.5, "Connected-component filter pass 2")
        filtered_mask2 = self.filter_small_regions(opened_mask, self.config.THRESHOLD2)
        results['filtered_mask2'] = filtered_mask2

        # Erosion
        if progress_callback: progress_callback(0.6, "Eroding filtered mask")
        kernel = np.ones((self.config.KERNEL2_SIZE, self.config.KERNEL2_SIZE), np.uint8)
        eroded_mask = cv2.erode(filtered_mask2, kernel, iterations=1)
        results['eroded_mask'] = eroded_mask

        # Connected-component filter pass 3
        if progress_callback: progress_callback(0.65, "Connected-component filter pass 3")
        filtered_mask3 = self.filter_small_regions(eroded_mask, self.config.THRESHOLD3)
        results['filtered_mask3'] = filtered_mask3

        # Enhanced thinning
        if progress_callback: progress_callback(0.7, "Enhanced thinning")
        thinned_mask = self.apply_enhanced_thinning(filtered_mask3)
        results['thinned_mask'] = thinned_mask

        # First enhanced burr removal
        if progress_callback: progress_callback(0.8, "Enhanced burr removal pass 1")
        deburr_mask1 = self.remove_enhanced_burr(thinned_mask)
        results['deburr_mask1'] = deburr_mask1

        # Second enhanced burr removal
        if progress_callback: progress_callback(0.85, "Enhanced burr removal pass 2")
        deburr_mask2 = self.remove_enhanced_burr(deburr_mask1)
        results['deburr_mask2'] = deburr_mask2

        # Remove tiny contours
        if progress_callback: progress_callback(0.9, "Removing small contours")
        final_mask = self.filter_small_regions(deburr_mask2, self.config.SMALL_CONTOUR_THRESHOLD)
        results['final_mask'] = final_mask

        return results

    def extract_contour_regions(self, final_mask: np.ndarray, original_img_info,
                                original_array: np.ndarray, dem_processor: DEMProcessor,
                                output_dir: str, progress_callback: Optional[Callable] = None) -> Tuple[List, List]:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(final_mask, connectivity=8)

        total_contours = num_labels - 1
        logger.info(f"Detected {total_contours} contour regions")

        # Determine resume point if previous batches exist
        resume_from, existing_stats_files, existing_pos_files = check_resume_point(output_dir)

        if resume_from > 1:
            logger.info(f"Resume mode: starting from contour {resume_from} (skipping first {resume_from - 1})")
            if progress_callback:
                progress_callback(0.9, f"Resume mode: starting at contour {resume_from}")

        # Batch parameters
        batch_size = 1500
        current_batch_stats = []
        current_batch_positions = []
        processed_count = 0

        # Iterate through target contours
        for i in range(resume_from, num_labels):
            # Update progress every 50 contours
            if progress_callback and (i - resume_from + 1) % 50 == 0:
                progress = 0.9 + ((i - resume_from + 1) / (total_contours - resume_from + 1)) * 0.1
                progress_callback(progress, f"Contour {i}/{total_contours}")

            try:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]

                current_batch_positions.append([i, x, y])

                # Build a square crop around the contour
                max_dimension = max(w, h)
                center_x = x + w // 2
                center_y = y + h // 2
                half_size = (max_dimension + 2 * self.config.PADDING) // 2

                square_x_min = max(0, center_x - half_size)
                square_y_min = max(0, center_y - half_size)
                square_x_max = min(original_img_info.width, center_x + half_size)
                square_y_max = min(original_img_info.height, center_y + half_size)

                actual_size = min(square_x_max - square_x_min, square_y_max - square_y_min)
                square_x_max = square_x_min + actual_size
                square_y_max = square_y_min + actual_size

                cropped_labels = labels[square_y_min:square_y_max, square_x_min:square_x_max]
                cropped_mask = (cropped_labels == i)
                edge_pixel_count = np.sum(cropped_mask)
                ratio_sum = (w / h) + (h / w) if w > 0 and h > 0 else 0

                # Contour descriptors
                region_mask = (labels == i)
                contours, _ = cv2.findContours(region_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    main_contour = max(contours, key=cv2.contourArea)
                    curvature_feature, roughness = self.calculate_contour_features(main_contour)
                else:
                    curvature_feature, roughness = 0.0, 1.0

                # Intensity and DEM statistics
                region_contour_mask = (labels == i) & (final_mask > 0)
                pixel_stats = self.calculate_pixel_statistics(region_contour_mask, original_array)
                dem_stats = dem_processor.calculate_dem_statistics_for_region(region_contour_mask)

                # Persist 1-bit preview PNG
                if cropped_mask.size > 0:
                    gray_bg = np.zeros((actual_size, actual_size), dtype=np.uint8)
                    copy_height = min(actual_size, cropped_mask.shape[0])
                    copy_width = min(actual_size, cropped_mask.shape[1])
                    if copy_height > 0 and copy_width > 0:
                        gray_bg[:copy_height, :copy_width] = cropped_mask[:copy_height, :copy_width] * 255
                    output_path = os.path.join(output_dir, f"{i}.png")
                    Image.fromarray(gray_bg, mode='L').save(output_path)

                # Collect stats row
                current_batch_stats.append([
                    i, edge_pixel_count, w, h, round(ratio_sum, 3),
                    curvature_feature, roughness,
                    pixel_stats['mean'], pixel_stats['std'], pixel_stats['median'],
                    dem_stats['mean'], dem_stats['std'], dem_stats['median'],
                    dem_stats['min'], dem_stats['max'], dem_stats['count'], ""
                ])

                processed_count += 1

                # Persist every batch_size contours
                if processed_count % batch_size == 0:
                    batch_start = i - batch_size + 1
                    if save_batch_csv(current_batch_stats, current_batch_positions, output_dir, batch_start,
                                      batch_size):
                        logger.info(f"Saved batch: contours {batch_start}-{i} ({batch_size} items)")
                        if progress_callback:
                            progress_callback(0.9 + ((i - resume_from + 1) / (total_contours - resume_from + 1)) * 0.1,
                                              f"Batch saved {batch_start}-{i}")

                    # Reset batch containers and free memory
                    current_batch_stats = []
                    current_batch_positions = []
                    gc.collect()  # Force GC to limit peak memory

            except Exception as e:
                logger.warning(f"Failed to process contour {i}: {e}")
                continue

        # Flush final partial batch
        if current_batch_stats:
            batch_start = total_contours - len(current_batch_stats) + 1
            if save_batch_csv(current_batch_stats, current_batch_positions, output_dir, batch_start):
                logger.info(f"Saved final batch: contours {batch_start}-{total_contours} ({len(current_batch_stats)} items)")

        # Merge every segment into a single file group
        final_saved_files = merge_all_csv_files(output_dir)

        # Final progress update
        if progress_callback:
            progress_callback(1.0, f"Contours processed (total {total_contours}, newly handled {processed_count})")

        logger.info(f"Contour extraction finished: total {total_contours}, newly handled {processed_count}")

        # Return nothing because CSV persistence already handled output
        return [], []


def check_resume_point(output_dir: str) -> Tuple[int, List, List]:
    """Determine resume information from previously saved batches."""
    try:
        stats_files = []
        pos_files = []

        # Scan the directory for batch fragments
        if not os.path.exists(output_dir):
            logger.info("Output directory missing; starting from scratch")
            return 1, [], []

        # Collect stats files
        all_files = os.listdir(output_dir)
        stats_pattern = []
        pos_pattern = []

        for filename in all_files:
            if filename.startswith('contour_statistics_') and filename.endswith('.csv'):
                # Extract numeric range, e.g., contour_statistics_15001_20000.csv
                try:
                    # Remove prefix and suffix
                    range_part = filename.replace('contour_statistics_', '').replace('.csv', '')
                    # Split start and end IDs
                    start_num, end_num = map(int, range_part.split('_'))
                    stats_pattern.append((start_num, end_num, filename))
                except:
                    continue

        for filename in all_files:
            if filename.startswith('contour_positions_') and filename.endswith('.csv'):
                try:
                    range_part = filename.replace('contour_positions_', '').replace('.csv', '')
                    start_num, end_num = map(int, range_part.split('_'))
                    pos_pattern.append((start_num, end_num, filename))
                except:
                    continue

        # Sort by starting index
        stats_pattern.sort(key=lambda x: x[0])
        pos_pattern.sort(key=lambda x: x[0])

        logger.info(f"Found {len(stats_pattern)} statistics fragments")
        logger.info(f"Found {len(pos_pattern)} position fragments")

        # Verify completeness of each fragment pair
        valid_segments = []
        for i, (start_num, end_num, stats_file) in enumerate(stats_pattern):
            # Find matching position file
            matching_pos = None
            for pos_start, pos_end, pos_file in pos_pattern:
                if pos_start == start_num and pos_end == end_num:
                    matching_pos = pos_file
                    break

            if matching_pos:
                stats_path = os.path.join(output_dir, stats_file)
                pos_path = os.path.join(output_dir, matching_pos)

                # Ensure both files exist and are non-empty
                if (os.path.exists(stats_path) and os.path.exists(pos_path) and
                        os.path.getsize(stats_path) > 0 and os.path.getsize(pos_path) > 0):
                    valid_segments.append((start_num, end_num, stats_file, matching_pos))
                    logger.info(f"Valid fragment: {start_num}-{end_num}")

        if not valid_segments:
            logger.info("No valid fragments found; starting from scratch")
            return 1, [], []

        # Use the largest end ID as the resume point
        max_end = max(valid_segments, key=lambda x: x[1])[1]
        resume_from = max_end + 1

        # Format lists for later merging
        stats_files = [os.path.join(output_dir, seg[2]) for seg in valid_segments]
        pos_files = [os.path.join(output_dir, seg[3]) for seg in valid_segments]

        logger.info(f"Detected {len(valid_segments)} valid fragments")
        logger.info(f"Fragments cover contours up to: {max_end}")
        logger.info(f"Resuming from contour {resume_from}")

        # Surface the first three fragments for quick inspection
        for start_num, end_num, stats_file, pos_file in valid_segments[:3]:
            logger.info(f"  Fragment: {start_num}-{end_num} (stats:{stats_file}, positions:{pos_file})")
        if len(valid_segments) > 3:
            logger.info(f"  ... {len(valid_segments) - 3} more fragments")

        return resume_from, stats_files, pos_files

    except Exception as e:
        logger.error(f"Failed to inspect resume information: {e}")
        logger.error(f"Stack trace: {traceback.format_exc()}")
        return 1, [], []


def save_batch_csv(statistics_data: List, position_data: List, output_dir: str,
                   start_idx: int, batch_size: int = 5000) -> bool:
    """Persist batch statistics and positions to CSV."""
    try:
        if not statistics_data or not position_data:
            return False

        end_idx = start_idx + len(statistics_data) - 1

        # Statistics CSV
        stats_filename = f"contour_statistics_{start_idx:05d}_{end_idx:05d}.csv"
        stats_path = os.path.join(output_dir, stats_filename)

        stats_columns = [
            'PNG Index', 'Edge Pixel Count', 'Bounding Box Length(px)', 'Bounding Box Width(px)',
            'L/W + W/L', 'Curvature Feature', 'Roughness', 'Pixel Mean', 'Pixel Std', 'Pixel Median',
            'DEM Mean', 'DEM Std', 'DEM Median', 'DEM Min', 'DEM Max', 'Valid DEM Pixels', 'Class'
        ]

        with open(stats_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(stats_columns)
            writer.writerows(statistics_data)

        # Position CSV
        pos_filename = f"contour_positions_{start_idx:05d}_{end_idx:05d}.csv"
        pos_path = os.path.join(output_dir, pos_filename)

        pos_columns = ['PNG Index', 'Top-Left X', 'Top-Left Y']

        with open(pos_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(pos_columns)
            writer.writerows(position_data)

        logger.info(f"Saved fragment files: {stats_filename}, {pos_filename}")
        return True

    except Exception as e:
        logger.error(f"Failed to persist batch CSV: {e}")
        return False


def merge_all_csv_files(output_dir: str) -> List[str]:
    """Merge every fragment CSV into consolidated outputs."""
    try:
        saved_files = []

        # Merge statistics files
        stats_files = sorted(
            [f for f in os.listdir(output_dir) if f.startswith('contour_statistics_') and f.endswith('.csv')])
        if stats_files:
            merged_stats_path = os.path.join(output_dir, "contour_statistics_enhanced_with_dem.csv")

            with open(merged_stats_path, 'w', newline='', encoding='utf-8-sig') as outfile:
                writer = csv.writer(outfile)
                header_written = False

                for stats_file in stats_files:
                    stats_file_path = os.path.join(output_dir, stats_file)
                    with open(stats_file_path, 'r', encoding='utf-8-sig') as infile:
                        reader = csv.reader(infile)
                        if not header_written:
                            writer.writerow(next(reader))
                            header_written = True
                        else:
                            next(reader)

                        for row in reader:
                            writer.writerow(row)

            saved_files.append("contour_statistics_enhanced_with_dem.csv")
            logger.info(f"Statistics merge complete: {len(stats_files)} fragments")

        # Merge position files
        pos_files = sorted(
            [f for f in os.listdir(output_dir) if f.startswith('contour_positions_') and f.endswith('.csv')])
        if pos_files:
            merged_pos_path = os.path.join(output_dir, "contour_positions.csv")

            with open(merged_pos_path, 'w', newline='', encoding='utf-8-sig') as outfile:
                writer = csv.writer(outfile)
                header_written = False

                for pos_file in pos_files:
                    pos_file_path = os.path.join(output_dir, pos_file)
                    with open(pos_file_path, 'r', encoding='utf-8-sig') as infile:
                        reader = csv.reader(infile)
                        if not header_written:
                            writer.writerow(next(reader))
                            header_written = True
                        else:
                            next(reader)

                        for row in reader:
                            writer.writerow(row)

            saved_files.append("contour_positions.csv")
            logger.info(f"Position merge complete: {len(pos_files)} fragments")

        return saved_files

    except Exception as e:
        logger.error(f"Failed to merge CSV fragments: {e}")
        return []


def save_statistics(statistics_data: List, position_data: List, output_dir: str) -> List[str]:
    """Persist statistics/position CSV files (simplified helper)."""
    saved_files = []
    os.makedirs(output_dir, exist_ok=True)

    # Statistics columns
    stats_columns = [
        'PNG Index', 'Edge Pixel Count', 'Bounding Box Length(px)', 'Bounding Box Width(px)',
        'L/W + W/L', 'Curvature Feature', 'Roughness', 'Pixel Mean', 'Pixel Std', 'Pixel Median',
        'DEM Mean', 'DEM Std', 'DEM Median', 'DEM Min', 'DEM Max', 'Valid DEM Pixels', 'Class'
    ]

    position_columns = ['PNG Index', 'Top-Left X', 'Top-Left Y']

    # Save statistics CSV
    if statistics_data:
        csv_path = os.path.join(output_dir, "contour_statistics_enhanced_with_dem.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(stats_columns)
            writer.writerows(statistics_data)
        saved_files.append("contour_statistics_enhanced_with_dem.csv")
        logger.info(f"Saved statistics CSV: {csv_path}")

    # Save position CSV
    if position_data:
        csv_path = os.path.join(output_dir, "contour_positions.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(position_columns)
            writer.writerows(position_data)
        saved_files.append("contour_positions.csv")
        logger.info(f"Saved position CSV: {csv_path}")

    return saved_files


def find_matching_files(sar_dir: str, dem_dir: str) -> List[Tuple[str, str]]:
    sar_files = {}
    dem_files = {}

    # Collect candidate files
    for ext in ['*.tif', '*.tiff']:
        for file_path in Path(sar_dir).glob(ext):
            sar_files[file_path.stem] = str(file_path)
        for file_path in Path(dem_dir).glob(ext):
            dem_files[file_path.stem] = str(file_path)

    # Identify matching SAR/DEM pairs
    matches = []
    for sar_base_name, sar_path in sar_files.items():
        expected_dem_base_name = sar_base_name + "_dem_processed"
        if expected_dem_base_name in dem_files:
            matches.append((sar_path, dem_files[expected_dem_base_name]))

    return matches


def process_single_file_worker(args):
    """Worker entry point with progress feedback."""
    sar_path, dem_path, output_dir, config_dict, progress_queue, process_slot = args

    config = LakeDetectionConfig()
    config.update_from_dict(config_dict)

    def progress_callback(progress, status):
        try:
            file_name = os.path.basename(sar_path)
            # Push progress updates into the queue
            progress_queue.put({
                'process_slot': process_slot,
                'progress': int(progress * 100),
                'status': f"🔄 {status} - {file_name}",
                'file_name': file_name
            })
        except:
            pass

    try:
        file_name = os.path.basename(sar_path)

        # Signal processing start
        progress_queue.put({
            'process_slot': process_slot,
            'progress': 0,
            'status': f"🚀 Processing started - {file_name}",
            'file_name': file_name
        })

        # Perform processing
        result = process_single_file(sar_path, dem_path, output_dir, config, progress_callback)

        # Signal completion
        if result['status'] == 'success':
            progress_queue.put({
                'process_slot': process_slot,
                'progress': 100,
                'status': f"✅ Completed - {file_name} ({result['contour_count']} contours)",
                'file_name': file_name
            })
        else:
            progress_queue.put({
                'process_slot': process_slot,
                'progress': 0,
                'status': f"❌ Failed - {file_name}",
                'file_name': file_name
            })

        return result

    except Exception as e:
        file_name = os.path.basename(sar_path)
        progress_queue.put({
            'process_slot': process_slot,
            'progress': 0,
            'status': f"💥 Exception - {file_name}: {str(e)[:50]}",
            'file_name': file_name
        })

        return {
            'status': 'error',
            'error': str(e),
            'output_dir': output_dir,
            'contour_count': 0,
            'statistics_files_saved': []
        }


def process_with_slot_wrapper(args):
    """Module-level wrapper that manages process slots to avoid pickle issues."""
    sar_path, dem_path, output_dir, config_dict, progress_queue, process_slot_queue = args

    # Acquire a process slot
    try:
        process_slot = process_slot_queue.get(timeout=60)
    except:
        process_slot = 0

    try:
        # Validate inputs one last time
        if not os.path.exists(sar_path):
            raise FileNotFoundError(f"SAR file not found: {sar_path}")
        if not os.path.exists(dem_path):
            raise FileNotFoundError(f"DEM file not found: {dem_path}")

        # Process using the acquired slot
        new_args = (sar_path, dem_path, output_dir, config_dict, progress_queue, process_slot)
        result = process_single_file_worker(new_args)
        return result

    except Exception as e:
        # Report failure through the progress queue
        file_name = os.path.basename(sar_path)
        try:
            progress_queue.put({
                'process_slot': process_slot,
                'progress': 0,
                'status': f"💥 Failed to launch - {file_name}: {str(e)[:50]}",
                'file_name': file_name
            })
        except:
            pass

        return {
            'status': 'error',
            'error': str(e),
            'output_dir': output_dir,
            'contour_count': 0,
            'statistics_files_saved': []
        }
    finally:
        # Release the slot
        try:
            process_slot_queue.put(process_slot)
        except:
            pass


def process_single_file(sar_path: str, dem_path: str, output_dir: str,
                        config: LakeDetectionConfig,
                        progress_callback: Optional[Callable] = None) -> Dict:
    try:
        os.makedirs(output_dir, exist_ok=True)

        processor = LakeDetectionProcessor(config)
        processor.read_geotiff_info(sar_path)
        dem_processor = DEMProcessor(dem_path, sar_path)

        # Read SAR raster with rasterio
        with rasterio.open(sar_path) as src:
            img_array = src.read(1)
            height, width = img_array.shape

            if img_array.dtype != np.uint8:
                img_min, img_max = np.min(img_array), np.max(img_array)
                if img_max != img_min:
                    img_array = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img_array = np.zeros_like(img_array, dtype=np.uint8)

        class ImageInfo:
            def __init__(self, width, height):
                self.width = width
                self.height = height

        original_img_info = ImageInfo(width, height)

        if progress_callback:
            progress_callback(0.05, "Starting image processing...")

        # Run the processing pipeline
        results = processor.process_image(img_array, progress_callback)

        if progress_callback:
            progress_callback(0.95, "Extracting contours...")

        # Extract contours and persist to disk (no in-memory return)
        _, _ = processor.extract_contour_regions(
            results['final_mask'], original_img_info, img_array, dem_processor, output_dir, progress_callback
        )

        # Count contours based on saved CSV
        contour_count = 0
        try:
            stats_csv = os.path.join(output_dir, "contour_statistics_enhanced_with_dem.csv")
            if os.path.exists(stats_csv):
                with open(stats_csv, 'r', encoding='utf-8-sig') as f:
                    reader = csv.reader(f)
                    next(reader)
                    contour_count = sum(1 for row in reader)
        except:
            pass

        # Save final outputs for visualization
        try:
            original_img_rgb = np.stack([img_array, img_array, img_array], axis=-1)
            contours_only_img = original_img_rgb.copy()
            contour_pixels = results['final_mask'] > 0
            contours_only_img[contour_pixels] = [255, 0, 0]

            contours_only_path = os.path.join(output_dir, "contours_only_enhanced_dem.tif")
            processor.save_array_as_geotiff(contours_only_img, contours_only_path)

            mask_only_path = os.path.join(output_dir, "contours_mask_only.tif")
            processor.save_array_as_geotiff(results['final_mask'], mask_only_path)

            # Save step-7 output (connected-component pass 3)
            step7_path = os.path.join(output_dir, "step7_filtered_mask3.tif")
            processor.save_array_as_geotiff(results['filtered_mask3'], step7_path)

            # Save final mask (after tiny contour removal)
            final_result_path = os.path.join(output_dir, "final_result_mask.tif")
            processor.save_array_as_geotiff(results['final_mask'], final_result_path)

            logger.info("Saved step-7 and final result rasters")

        except Exception as e:
            logger.warning(f"Failed to save final imagery: {e}")

        if progress_callback:
            progress_callback(1.0, "Processing complete")

        return {
            'status': 'success',
            'contour_count': contour_count,
            'output_dir': output_dir,
            'statistics_files_saved': ["contour_statistics_enhanced_with_dem.csv", "contour_positions.csv"]
        }

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'output_dir': output_dir,
            'contour_count': 0,
            'statistics_files_saved': []
        }


class ProcessingThread(QThread):
    progress_updated = pyqtSignal(int, str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str, str)
    batch_progress_updated = pyqtSignal(int, int, str)

    def __init__(self, task_type, **kwargs):
        super().__init__()
        self.task_type = task_type
        self.kwargs = kwargs

    def run(self):
        try:
            if self.task_type == 'single':
                result = self.process_single_file()
            elif self.task_type == 'batch':
                result = self.process_batch_files()
            self.finished_signal.emit(result)
        except Exception as e:
            self.error_signal.emit("Processing error", str(e))

    def progress_callback(self, progress, status):
        self.progress_updated.emit(int(progress * 100), status)

    def batch_progress_callback(self, process_id, progress, status):
        self.batch_progress_updated.emit(process_id, progress, status)

    def process_single_file(self):
        return process_single_file(
            self.kwargs['sar_path'],
            self.kwargs['dem_path'],
            self.kwargs['output_dir'],
            self.kwargs['config'],
            self.progress_callback
        )

    def process_batch_files(self):
        try:
            sar_dir = self.kwargs['sar_dir']
            dem_dir = self.kwargs['dem_dir']
            output_dir = self.kwargs['output_dir']
            config = self.kwargs['config']

            # Detailed logging for batch execution
            logger.info("Starting batch processing...")
            logger.info(f"SAR directory: {sar_dir}")
            logger.info(f"DEM directory: {dem_dir}")
            logger.info(f"Output directory: {output_dir}")

            # Validate input directories
            if not os.path.exists(sar_dir):
                error_msg = f"SAR directory does not exist: {sar_dir}"
                logger.error(error_msg)
                return {'status': 'error', 'error': error_msg}

            if not os.path.exists(dem_dir):
                error_msg = f"DEM directory does not exist: {dem_dir}"
                logger.error(error_msg)
                return {'status': 'error', 'error': error_msg}

            file_pairs = find_matching_files(sar_dir, dem_dir)
            logger.info(f"Found {len(file_pairs)} matching file pairs")

            if not file_pairs:
                # Provide detailed diagnostics when no matches are found
                sar_files = []
                dem_files = []

                for ext in ['*.tif', '*.tiff']:
                    sar_files.extend([str(f) for f in Path(sar_dir).glob(ext)])
                    dem_files.extend([str(f) for f in Path(dem_dir).glob(ext)])

                error_msg = "No matching file pairs were found!\n"
                error_msg += f"{len(sar_files)} SAR files detected:\n"
                for f in sar_files[:5]:
                    error_msg += f"  - {os.path.basename(f)}\n"
                if len(sar_files) > 5:
                    error_msg += f"  ... {len(sar_files) - 5} more files\n"

                error_msg += f"{len(dem_files)} DEM files detected:\n"
                for f in dem_files[:5]:
                    error_msg += f"  - {os.path.basename(f)}\n"
                if len(dem_files) > 5:
                    error_msg += f"  ... {len(dem_files) - 5} more files\n"

                error_msg += "\nMatching rule: <sar_name>.tif pairs with <sar_name>_dem_processed.tif"

                logger.error(error_msg)
                return {'status': 'error', 'error': error_msg}

            os.makedirs(output_dir, exist_ok=True)

            # Preview the first few matches
            logger.info("Matching file pairs:")
            for i, (sar_path, dem_path) in enumerate(file_pairs[:5]):
                logger.info(f"  {i + 1}. SAR: {os.path.basename(sar_path)}")
                logger.info(f"     DEM: {os.path.basename(dem_path)}")
            if len(file_pairs) > 5:
                logger.info(f"  ... {len(file_pairs) - 5} more pairs")

            # Shared queue for progress updates
            manager = Manager()
            progress_queue = manager.Queue()

            # Prepare worker arguments (simple variant without slot queue)
            process_args = []
            for idx, (sar_path, dem_path) in enumerate(file_pairs):
                # Validate file presence
                if not os.path.exists(sar_path):
                    logger.error(f"SAR file missing: {sar_path}")
                    continue
                if not os.path.exists(dem_path):
                    logger.error(f"DEM file missing: {dem_path}")
                    continue

                file_name = Path(sar_path).stem
                file_output_dir = os.path.join(output_dir, file_name)

                # Ensure output directory exists
                os.makedirs(file_output_dir, exist_ok=True)

                # Cycle process slot IDs between 0-7
                process_slot = idx % 8

                # Pass process slot directly
                process_args.append(
                    (sar_path, dem_path, file_output_dir, config.to_dict(), progress_queue, process_slot))

            if not process_args:
                error_msg = "All candidate files failed validation; nothing to process"
                logger.error(error_msg)
                return {'status': 'error', 'error': error_msg}

            logger.info(f"Preparing to process {len(process_args)} files")

            batch_results = []
            successful_count = 0
            failed_count = 0

            # Start progress monitoring thread
            monitoring_active = True

            def monitor_progress():
                while monitoring_active:
                    try:
                        # Pull updates without blocking
                        while not progress_queue.empty():
                            try:
                                progress_info = progress_queue.get_nowait()
                                process_slot = progress_info.get('process_slot', 0)
                                progress = progress_info.get('progress', 0)
                                status = progress_info.get('status', 'Idle')

                                if hasattr(self, 'batch_progress_callback') and 0 <= process_slot < 8:
                                    self.batch_progress_callback(process_slot, progress, status)
                            except:
                                break
                        time.sleep(0.1)
                    except Exception as e:
                        logger.warning(f"Progress monitor failed: {e}")
                        break

            monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
            monitor_thread.start()

            # Launch an 8-worker pool
            logger.info("Spawning 8-process pool...")

            with mp.Pool(processes=8) as pool:
                # Submit all tasks
                futures = []
                for i, args in enumerate(process_args):
                    try:
                        # Use the existing worker function
                        future = pool.apply_async(process_single_file_worker, (args,))
                        futures.append((future, args, i))
                        logger.info(f"Submitted task {i + 1}: {os.path.basename(args[0])}")
                    except Exception as e:
                        logger.error(f"Failed to submit task {i + 1}: {e}")
                        failed_count += 1

                logger.info(f"Submitted {len(futures)} tasks to the pool")

                # Gather results
                for i, (future, args, task_id) in enumerate(futures):
                    main_progress = (i + 1) / len(futures)
                    if hasattr(self, 'progress_callback'):
                        self.progress_callback(main_progress, f"Task {i + 1}/{len(futures)} complete")

                    try:
                        logger.info(f"Waiting for task {task_id + 1} ...")
                        result = future.get(timeout=3600)

                        file_name = Path(args[0]).stem
                        result['file_name'] = file_name
                        batch_results.append(result)

                        if result['status'] == 'success':
                            successful_count += 1
                            logger.info(f"Task {task_id + 1} completed successfully: {file_name}")
                        else:
                            failed_count += 1
                            logger.warning(
                                f"Task {task_id + 1} failed: {file_name} - {result.get('error', 'Unknown error')}")

                    except Exception as e:
                        failed_count += 1
                        file_name = Path(args[0]).stem
                        logger.error(f"Task {task_id + 1} raised an exception: {file_name} - {e}")
                        batch_results.append({
                            'status': 'error',
                            'file_name': file_name,
                            'error': str(e),
                            'contour_count': 0
                        })

            # Stop monitoring
            monitoring_active = False
            if monitor_thread.is_alive():
                monitor_thread.join(timeout=1)

            # Reset UI progress state
            for i in range(8):
                if hasattr(self, 'batch_progress_callback'):
                    self.batch_progress_callback(i, 0, f"Process {i + 1}: Idle")

            # Persist batch report
            report = {
                'status': 'completed',
                'total_files': len(file_pairs),
                'successful_count': successful_count,
                'failed_count': failed_count,
                'output_dir': output_dir,
                'file_results': batch_results
            }

            report_path = os.path.join(output_dir, 'batch_processing_report.json')
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            logger.info(f"Batch processing complete: {successful_count} success, {failed_count} failure")

            if hasattr(self, 'progress_callback'):
                self.progress_callback(1.0, f"Batch complete: {successful_count} success, {failed_count} failure")

            return report

        except Exception as e:
            logger.error(f"Batch processing crashed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {'status': 'error', 'error': str(e)}


class LakeDetectionMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = LakeDetectionConfig()
        self.settings = QSettings('LakeDetection', 'MainApp')
        self.processing_thread = None
        self.current_batch_status = {}  # Track batch-processing status
        self.init_ui()
        self.load_settings()

    def init_ui(self):
        self.setWindowTitle("Lake Contour Extraction Tool v2.1 (Enhanced Thinning)")
        self.setGeometry(100, 100, 1000, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        self.create_parameter_panel(splitter)
        self.create_main_panel(splitter)

        splitter.setSizes([250, 750])
        self.statusBar().showMessage("Ready – Step4 enhanced thinning integrated")

    def create_parameter_panel(self, parent):
        param_widget = QWidget()
        param_layout = QVBoxLayout(param_widget)

        param_group = QGroupBox("Parameter Settings")
        param_form = QFormLayout(param_group)

        self.param_inputs = {}

        params = [
            ("Edge Detection Threshold", "EDGE_THRESHOLD", 1, 255),
            ("Connected Component Filter 1", "THRESHOLD1", 10, 1000),
            ("Opening Kernel Size", "KERNEL1_SIZE", 1, 20),
            ("Connected Component Filter 2", "THRESHOLD2", 10, 1000),
            ("Erosion Kernel Size", "KERNEL2_SIZE", 1, 20),
            ("Connected Component Filter 3", "THRESHOLD3", 10, 2000),
            ("Burr Removal Threshold", "BURR_THRESHOLD", 5, 100),
            ("Small Contour Threshold", "SMALL_CONTOUR_THRESHOLD", 5, 100),
            ("Padding", "PADDING", 0, 20),
        ]

        for label, param_name, min_val, max_val in params:
            spinbox = QSpinBox()
            spinbox.setRange(min_val, max_val)
            spinbox.setValue(getattr(self.config, param_name))
            spinbox.valueChanged.connect(self.update_config)
            self.param_inputs[param_name] = spinbox
            param_form.addRow(f"{label}:", spinbox)

        self.enable_dilation_cb = QCheckBox("Enable dilation repair")
        self.enable_dilation_cb.setChecked(self.config.ENABLE_DILATION)
        self.enable_dilation_cb.stateChanged.connect(self.update_config)
        param_form.addRow(self.enable_dilation_cb)

        param_layout.addWidget(param_group)

        # Enhanced feature summary
        features_group = QGroupBox("Enhanced Features")
        features_layout = QVBoxLayout(features_group)

        features_text = QLabel("""🚀 New capabilities:
• Vectorized endpoint detection
• Rotation/flip-aware pattern matching
• Dual-stage burr removal
• Zhang-Suen thinning
• Identical behavior to Step4""")
        features_text.setWordWrap(True)
        features_text.setStyleSheet("color: #007ACC; font-size: 10px;")
        features_layout.addWidget(features_text)

        param_layout.addWidget(features_group)
        param_layout.addStretch()
        parent.addWidget(param_widget)

    def create_main_panel(self, parent):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        self.create_single_file_tab()
        self.create_batch_processing_tab()
        self.create_results_tab()

        # Progress display
        progress_group = QGroupBox("Processing Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)

        self.status_label = QLabel("Ready – enhanced thinning is active")
        self.status_label.setAlignment(Qt.AlignCenter)
        progress_layout.addWidget(self.status_label)

        main_layout.addWidget(progress_group)
        parent.addWidget(main_widget)

    def create_single_file_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        file_group = QGroupBox("File Selection")
        file_layout = QFormLayout(file_group)

        # SAR file
        sar_layout = QHBoxLayout()
        self.sar_path_edit = QLineEdit()
        sar_browse_btn = QPushButton("Browse")
        sar_browse_btn.clicked.connect(self.browse_sar_file)
        sar_layout.addWidget(self.sar_path_edit)
        sar_layout.addWidget(sar_browse_btn)
        file_layout.addRow("SAR image:", sar_layout)

        # DEM file
        dem_layout = QHBoxLayout()
        self.dem_path_edit = QLineEdit()
        dem_browse_btn = QPushButton("Browse")
        dem_browse_btn.clicked.connect(self.browse_dem_file)
        dem_layout.addWidget(self.dem_path_edit)
        dem_layout.addWidget(dem_browse_btn)
        file_layout.addRow("DEM data:", dem_layout)

        # Output directory
        output_layout = QHBoxLayout()
        self.output_path_edit = QLineEdit()
        output_browse_btn = QPushButton("Browse")
        output_browse_btn.clicked.connect(self.browse_output_dir)
        output_layout.addWidget(self.output_path_edit)
        output_layout.addWidget(output_browse_btn)
        file_layout.addRow("Output directory:", output_layout)

        layout.addWidget(file_group)

        self.single_process_btn = QPushButton("Start processing (enhanced thinning)")
        self.single_process_btn.clicked.connect(self.start_single_processing)
        layout.addWidget(self.single_process_btn)

        layout.addStretch()
        self.tab_widget.addTab(tab, "Single file")

    def create_batch_processing_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        folder_group = QGroupBox("Folder Selection")
        folder_layout = QFormLayout(folder_group)

        # SAR folder
        sar_dir_layout = QHBoxLayout()
        self.sar_dir_edit = QLineEdit()
        sar_dir_browse_btn = QPushButton("Browse")
        sar_dir_browse_btn.clicked.connect(self.browse_sar_dir)
        sar_dir_layout.addWidget(self.sar_dir_edit)
        sar_dir_layout.addWidget(sar_dir_browse_btn)
        folder_layout.addRow("SAR image folder:", sar_dir_layout)

        # DEM folder
        dem_dir_layout = QHBoxLayout()
        self.dem_dir_edit = QLineEdit()
        dem_dir_browse_btn = QPushButton("Browse")
        dem_dir_browse_btn.clicked.connect(self.browse_dem_dir)
        dem_dir_layout.addWidget(self.dem_dir_edit)
        dem_dir_layout.addWidget(dem_dir_browse_btn)
        folder_layout.addRow("DEM data folder:", dem_dir_layout)

        # Output folder
        batch_output_layout = QHBoxLayout()
        self.batch_output_edit = QLineEdit()
        batch_output_browse_btn = QPushButton("Browse")
        batch_output_browse_btn.clicked.connect(self.browse_batch_output_dir)
        batch_output_layout.addWidget(self.batch_output_edit)
        batch_output_layout.addWidget(batch_output_browse_btn)
        folder_layout.addRow("Output folder:", batch_output_layout)

        layout.addWidget(folder_group)

        # Eight-process progress
        progress_group = QGroupBox("8-process progress (enhanced)")
        progress_layout = QVBoxLayout(progress_group)

        self.process_progress_bars = []
        self.process_status_labels = []

        for i in range(8):
            process_layout = QHBoxLayout()

            process_label = QLabel(f"Process {i + 1}:")
            process_label.setMinimumWidth(60)
            process_layout.addWidget(process_label)

            progress_bar = QProgressBar()
            progress_bar.setVisible(False)
            self.process_progress_bars.append(progress_bar)
            process_layout.addWidget(progress_bar)

            status_label = QLabel("Idle")
            status_label.setMinimumWidth(300)  # Extra width for filenames
            status_label.setWordWrap(True)  # Allow wrapping
            self.process_status_labels.append(status_label)
            process_layout.addWidget(status_label)

            progress_layout.addLayout(process_layout)

        layout.addWidget(progress_group)

        self.batch_process_btn = QPushButton("Start 8-process batch (enhanced)")
        self.batch_process_btn.clicked.connect(self.start_batch_processing)
        layout.addWidget(self.batch_process_btn)

        layout.addStretch()
        self.tab_widget.addTab(tab, "Batch processing")

    def create_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "File", "Status", "Contour count", "Output directory", "CSV files"
        ])

        header = self.results_table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeToContents)
        self.results_table.setAlternatingRowColors(True)

        layout.addWidget(self.results_table)

        button_layout = QHBoxLayout()
        clear_results_btn = QPushButton("Clear results")
        clear_results_btn.clicked.connect(self.clear_results)
        button_layout.addWidget(clear_results_btn)
        button_layout.addStretch()
        layout.addLayout(button_layout)

        self.tab_widget.addTab(tab, "Processing results")

    def browse_sar_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select SAR image", "", "TIFF Files (*.tif *.tiff)")
        if file_path:
            self.sar_path_edit.setText(file_path)

    def browse_dem_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select DEM file", "", "TIFF Files (*.tif *.tiff)")
        if file_path:
            self.dem_path_edit.setText(file_path)

    def browse_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select output directory")
        if dir_path:
            self.output_path_edit.setText(dir_path)

    def browse_sar_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select SAR image folder")
        if dir_path:
            self.sar_dir_edit.setText(dir_path)

    def browse_dem_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select DEM data folder")
        if dir_path:
            self.dem_dir_edit.setText(dir_path)

    def browse_batch_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select output folder")
        if dir_path:
            self.batch_output_edit.setText(dir_path)

    def update_config(self):
        for param_name, input_widget in self.param_inputs.items():
            setattr(self.config, param_name, int(input_widget.value()))
        self.config.ENABLE_DILATION = self.enable_dilation_cb.isChecked()

    def start_single_processing(self):
        if self.processing_thread and self.processing_thread.isRunning():
            QMessageBox.warning(self, "Processing", "A processing task is already running.")
            return

        sar_path = self.sar_path_edit.text().strip()
        dem_path = self.dem_path_edit.text().strip()
        output_dir = self.output_path_edit.text().strip()

        if not all([sar_path, dem_path, output_dir]):
            QMessageBox.warning(self, "Missing parameters", "Please select every required file and directory.")
            return

        self.update_config()

        self.processing_thread = ProcessingThread(
            'single',
            sar_path=sar_path,
            dem_path=dem_path,
            output_dir=output_dir,
            config=self.config
        )

        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.finished_signal.connect(self.on_single_processing_finished)
        self.processing_thread.error_signal.connect(self.on_processing_error)

        self.processing_thread.start()
        self.single_process_btn.setEnabled(False)
        self.batch_process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

    def start_batch_processing(self):
        if self.processing_thread and self.processing_thread.isRunning():
            QMessageBox.warning(self, "Processing", "A processing task is already running.")
            return

        sar_dir = self.sar_dir_edit.text().strip()
        dem_dir = self.dem_dir_edit.text().strip()
        output_dir = self.batch_output_edit.text().strip()

        if not all([sar_dir, dem_dir, output_dir]):
            QMessageBox.warning(self, "Missing parameters", "Please select every required folder.")
            return

        self.update_config()

        # Show per-process progress bars
        for i in range(8):
            self.process_progress_bars[i].setVisible(True)
            self.process_progress_bars[i].setValue(0)
            self.process_status_labels[i].setText(f"Process {i + 1}: ready for tasks...")

        # Reset batch-status tracking
        self.current_batch_status = {}
        for i in range(8):
            self.current_batch_status[i] = {
                'progress': 0,
                'status': f"Process {i + 1}: waiting for assignment...",
                'file': '',
                'active': False
            }

        self.processing_thread = ProcessingThread(
            'batch',
            sar_dir=sar_dir,
            dem_dir=dem_dir,
            output_dir=output_dir,
            config=self.config
        )

        self.processing_thread.progress_updated.connect(self.update_progress)
        self.processing_thread.batch_progress_updated.connect(self.update_batch_progress)
        self.processing_thread.finished_signal.connect(self.on_batch_processing_finished)
        self.processing_thread.error_signal.connect(self.on_processing_error)

        self.processing_thread.start()
        self.single_process_btn.setEnabled(False)
        self.batch_process_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

    @pyqtSlot(int, int, str)
    def update_batch_progress(self, process_id, progress, status):
        try:
            if 0 <= process_id < 8:
                # Clamp progress safely between 0-100
                progress_value = max(0, min(100, int(progress)))
                self.process_progress_bars[process_id].setValue(progress_value)
                self.process_status_labels[process_id].setText(str(status)[:200])  # Truncate status text

                # Adjust label color based on state
                status_str = str(status)
                if "✅" in status_str or "Completed" in status_str:
                    color_style = "QLabel { color: #4CAF50; font-size: 11px; font-weight: bold; }"
                elif "❌" in status_str or "Failed" in status_str:
                    color_style = "QLabel { color: #f44336; font-size: 11px; font-weight: bold; }"
                elif "💥" in status_str or "Exception" in status_str:
                    color_style = "QLabel { color: #ff9800; font-size: 11px; font-weight: bold; }"
                elif "🔄" in status_str or "Processing" in status_str or "Contour" in status_str:
                    color_style = "QLabel { color: #2196F3; font-size: 11px; }"
                else:
                    color_style = "QLabel { color: #666; font-size: 11px; }"

                self.process_status_labels[process_id].setStyleSheet(color_style)

        except Exception as e:
            logger.warning(f"Failed to update progress bar for process {process_id + 1}: {e}")

    @pyqtSlot(int, str)
    def update_progress(self, progress, status):
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)
        self.statusBar().showMessage(status)

    @pyqtSlot(dict)
    def on_single_processing_finished(self, result):
        self.processing_finished()

        if result['status'] == 'success':
            self.add_result_to_table(
                filename=os.path.basename(self.sar_path_edit.text()),
                status="Success",
                contour_count=result.get('contour_count', 0),
                output_dir=result.get('output_dir', ''),
                statistics_files=', '.join(result.get('statistics_files_saved', []))
            )
            QMessageBox.information(
                self,
                "Processing complete",
                f"Enhanced thinning succeeded!\nContours: {result['contour_count']}\n"
                f"Output directory: {result['output_dir']}\n\n"
                f"🚀 Features enabled:\n"
                f"✅ Vectorized endpoint detection\n"
                f"✅ Dual-stage burr removal\n"
                f"✅ Zhang-Suen thinning\n"
                f"✅ Resume support"
            )
            self.tab_widget.setCurrentIndex(2)
        else:
            QMessageBox.critical(self, "Processing failed", f"Processing failed: {result.get('error', 'Unknown error')}")

    @pyqtSlot(dict)
    def on_batch_processing_finished(self, result):
        self.processing_finished()

        if result['status'] == 'completed':
            for file_result in result['file_results']:
                self.add_result_to_table(
                    filename=file_result.get('file_name', 'Unknown file'),
                    status="Success" if file_result['status'] == 'success' else "Failed",
                    contour_count=file_result.get('contour_count', 0),
                    output_dir=file_result.get('output_dir', ''),
                    statistics_files=', '.join(file_result.get('statistics_files_saved', []))
                )

            QMessageBox.information(
                self, "Batch complete",
                f"Enhanced thinning batch completed!\nTotal files: {result['total_files']}\n"
                f"Success: {result['successful_count']}\nFailure: {result['failed_count']}\n\n"
                f"🚀 Enhanced pipeline enabled for all files:\n"
                f"✅ Vectorized endpoint detection\n"
                f"✅ Dual-stage burr removal\n"
                f"✅ Step4-equivalent workflow"
            )
            self.tab_widget.setCurrentIndex(2)
        else:
            QMessageBox.critical(self, "Batch failed", f"Batch failed: {result.get('error', 'Unknown error')}")

    @pyqtSlot(str, str)
    def on_processing_error(self, title, message):
        self.processing_finished()
        QMessageBox.critical(self, title, message)

    def processing_finished(self):
        self.single_process_btn.setEnabled(True)
        self.batch_process_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_label.setText("Ready – enhanced thinning is active")
        self.statusBar().showMessage("Ready – enhanced thinning is active")

        # Hide per-process bars and reset state
        for i in range(8):
            self.process_progress_bars[i].setVisible(False)
            self.process_status_labels[i].setText(f"Process {i + 1}: Idle")
            self.process_status_labels[i].setStyleSheet("QLabel { color: #666; font-size: 11px; }")

        # Clear batch-status tracking map
        if hasattr(self, 'current_batch_status'):
            self.current_batch_status.clear()

    def add_result_to_table(self, filename, status, contour_count, output_dir, statistics_files):
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)

        self.results_table.setItem(row, 0, QTableWidgetItem(str(filename)))
        self.results_table.setItem(row, 1, QTableWidgetItem(str(status)))
        self.results_table.setItem(row, 2, QTableWidgetItem(str(contour_count)))
        self.results_table.setItem(row, 3, QTableWidgetItem(str(output_dir)))
        self.results_table.setItem(row, 4, QTableWidgetItem(str(statistics_files)))

    def clear_results(self):
        reply = QMessageBox.question(self, "Clear results", "Clear every processing result?")
        if reply == QMessageBox.Yes:
            self.results_table.setRowCount(0)

    def save_settings(self):
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("sar_path", self.sar_path_edit.text())
        self.settings.setValue("dem_path", self.dem_path_edit.text())
        self.settings.setValue("output_path", self.output_path_edit.text())
        self.settings.setValue("sar_dir", self.sar_dir_edit.text())
        self.settings.setValue("dem_dir", self.dem_dir_edit.text())
        self.settings.setValue("batch_output", self.batch_output_edit.text())

    def load_settings(self):
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

        self.sar_path_edit.setText(self.settings.value("sar_path", ""))
        self.dem_path_edit.setText(self.settings.value("dem_path", ""))
        self.output_path_edit.setText(self.settings.value("output_path", ""))
        self.sar_dir_edit.setText(self.settings.value("sar_dir", ""))
        self.dem_dir_edit.setText(self.settings.value("dem_dir", ""))
        self.batch_output_edit.setText(self.settings.value("batch_output", ""))

    def closeEvent(self, event):
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self, "Processing", "A processing job is running. Exit anyway?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                event.ignore()
                return
            self.processing_thread.terminate()
            self.processing_thread.wait()

        self.save_settings()
        event.accept()


def main():
    if hasattr(mp, 'set_start_method'):
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    app = QApplication(sys.argv)
    app.setApplicationName("Lake Contour Extraction Tool")
    app.setApplicationVersion("2.1 - Enhanced Thinning")

    window = LakeDetectionMainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
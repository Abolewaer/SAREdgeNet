import sys
import os
import glob
import traceback
import time

from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget,
                             QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                             QPushButton, QTextEdit, QProgressBar, QFileDialog,
                             QDoubleSpinBox, QMessageBox, QGroupBox, QGridLayout,
                             QCheckBox, QSpacerItem, QSizePolicy, QSpinBox,
                             QFrame, QScrollArea, QComboBox, QRadioButton, QButtonGroup)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon

# Import the original processing utilities
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.plot import show
import matplotlib as mpl
from scipy.ndimage import gaussian_filter, median_filter, grey_erosion, grey_dilation
from rasterio.transform import Affine, rowcol
import rasterio.mask
from rasterio.warp import reproject, Resampling, calculate_default_transform
from shapely.geometry import box
import cv2
from scipy import ndimage
import gc
import csv
import time

# Configure matplotlib to support Chinese fonts
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# Try to import optional dependencies
try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from osgeo import gdal, osr

    GDAL_AVAILABLE = True
except ImportError:
    GDAL_AVAILABLE = False


class Step1Processor:
    """Step 1 processor"""

    @staticmethod
    def process_single_tif(tif_path, output_path=None):
        """Process a single TIF file"""
        try:
            with rasterio.open(tif_path) as src:
                profile = src.profile.copy()
                transform = src.transform
                crs = src.crs

                img = src.read(1).astype(np.float32)

                non_zero_mask = img != 0 
                non_zero_pixels = img[non_zero_mask]

                if len(non_zero_pixels) == 0:
                    return False, "No non-zero pixels found in the image!"

                mean_val = float(np.mean(non_zero_pixels))
                std_val = float(np.std(non_zero_pixels))
                threshold = mean_val + 2 * std_val

                processed_img = np.copy(img)
                mask = (processed_img != 0) & (processed_img > threshold)
                processed_img[mask] = threshold

                normalized_img = np.zeros_like(processed_img, dtype=np.uint8)

                if np.any(non_zero_mask):
                    min_val = float(np.min(processed_img[non_zero_mask]))
                    max_val = float(np.max(processed_img[non_zero_mask]))

                    if max_val > min_val:
                        normalized_img[non_zero_mask] = np.clip(
                            (255 * (processed_img[non_zero_mask] - min_val) / (max_val - min_val)).astype(np.uint8),
                            0, 255
                        )

                del processed_img

                sigma = 1.0
                gaussian_img = gaussian_filter(normalized_img, sigma=sigma).astype(np.uint8)
                median_img = median_filter(gaussian_img, size=3).astype(np.uint8)
                del gaussian_img

                structure_size = 3
                eroded_img = grey_erosion(median_img, size=(structure_size, structure_size)).astype(np.uint8)
                del median_img

                dilated_img = grey_dilation(eroded_img, size=(structure_size, structure_size)).astype(np.uint8)
                del eroded_img

                if output_path:
                    profile.update(
                        dtype=rasterio.uint8,
                        count=1,
                        compress='lzw',
                        crs=crs,
                        transform=transform
                    )

                    with rasterio.open(output_path, 'w', **profile) as dst:
                        dst.write(dilated_img, 1)

                return True, f"Processing complete: {os.path.basename(tif_path)}"

        except Exception as e:
            return False, f"Processing failed: {str(e)}"


class Step2Processor:
    """Step 2 processor"""

    def __init__(self):
        self.original_data = None
        self.original_profile = None
        self.original_transform = None
        self.original_bounds = None
        self.adjusted_data = None
        self.adjusted_profile = None
        self.adjusted_transform = None
        self.adjusted_bounds = None

    def load_sar_image(self, sar_path):
        try:
            with rasterio.open(sar_path) as src:
                self.original_data = src.read(1)
                self.original_profile = src.profile.copy()
                self.original_transform = src.transform
                self.original_bounds = src.bounds
                self.original_crs = src.crs
                return True, "SAR image loaded successfully"
        except Exception as e:
            return False, f"Failed to load SAR image: {e}"

    def adjust_geolocation(self, delta_lon, delta_lat):
        try:
            if self.original_transform is None:
                return False, "Load the SAR image first!"

            original_transform = self.original_transform
            new_x_origin = original_transform[2] + delta_lon
            new_y_origin = original_transform[5] + delta_lat

            self.adjusted_transform = Affine(
                original_transform[0],
                original_transform[1],
                new_x_origin,
                original_transform[3],
                original_transform[4],
                new_y_origin
            )

            self.adjusted_data = self.original_data.copy()
            self.adjusted_profile = self.original_profile.copy()
            self.adjusted_profile.update({
                'transform': self.adjusted_transform
            })

            height, width = self.original_data.shape
            new_left = new_x_origin
            new_top = new_y_origin
            new_right = new_left + width * original_transform[0]
            new_bottom = new_top + height * original_transform[4]

            from rasterio.coords import BoundingBox
            self.adjusted_bounds = BoundingBox(new_left, new_bottom, new_right, new_top)

            return True, "Geolocation adjustment complete"

        except Exception as e:
            return False, f"Failed to adjust geolocation: {e}"

    def save_adjusted_sar(self, output_path):
        try:
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            with rasterio.open(output_path, 'w', **self.adjusted_profile) as dst:
                dst.write(self.adjusted_data, 1)

            return True, f"Adjusted SAR image saved: {output_path}"

        except Exception as e:
            return False, f"Failed to save: {e}"


class Step3Processor:
    """Step 3 processor - supports single file and batch processing"""

    def __init__(self):
        self.dem_data = None
        self.dem_profile = None
        self.reference_bounds = None
        self.processed_data = None

    def load_reference_bounds(self, reference_tif_path):
        try:
            with rasterio.open(reference_tif_path) as src:
                self.reference_bounds = src.bounds
                self.reference_crs = src.crs
                self.reference_transform = src.transform
                self.reference_shape = (src.height, src.width)
                return True, "Reference file loaded successfully"
        except Exception as e:
            return False, f"Failed to load reference file: {e}"

    def load_dem(self, dem_path):
        try:
            with rasterio.open(dem_path) as src:
                self.dem_profile = src.profile.copy()
                self.dem_data = src.read(1)
                self.dem_bounds = src.bounds
                self.dem_crs = src.crs
                self.dem_transform = src.transform
                return True, "DEM file loaded successfully"
        except Exception as e:
            return False, f"Failed to load DEM file: {e}"

    def crop_dem_to_reference(self):
        try:
            dem_transform = self.dem_transform

            def geo_to_pixel(geo_x, geo_y, transform):
                row, col = rowcol(transform, geo_x, geo_y)
                return int(col), int(row)

            left_px, top_px = geo_to_pixel(self.reference_bounds.left, self.reference_bounds.top, dem_transform)
            right_px, bottom_px = geo_to_pixel(self.reference_bounds.right, self.reference_bounds.bottom, dem_transform)

            height, width = self.dem_data.shape
            left_px = max(0, min(left_px, width - 1))
            right_px = max(0, min(right_px, width - 1))
            top_px = max(0, min(top_px, height - 1))
            bottom_px = max(0, min(bottom_px, height - 1))

            if left_px > right_px:
                left_px, right_px = right_px, left_px
            if top_px > bottom_px:
                top_px, bottom_px = bottom_px, top_px

            self.cropped_dem = self.dem_data[top_px:bottom_px + 1, left_px:right_px + 1].copy()

            new_left = dem_transform[2] + left_px * dem_transform[0]
            new_top = dem_transform[5] + top_px * dem_transform[4]

            self.cropped_transform = Affine(
                dem_transform[0],
                dem_transform[1],
                new_left,
                dem_transform[3],
                dem_transform[4],
                new_top
            )

            return True, "DEM cropping complete"

        except Exception as e:
            return False, f"Failed to crop DEM: {e}"

    def preprocess_dem_2sigma(self):
        try:
            nodata_candidates = [0, -9999, -32768, 32767]
            nodata_value = None

            for candidate in nodata_candidates:
                if np.any(self.cropped_dem == candidate):
                    nodata_value = candidate
                    break

            if nodata_value is not None:
                valid_mask = self.cropped_dem != nodata_value
            else:
                valid_mask = np.ones_like(self.cropped_dem, dtype=bool)

            valid_data = self.cropped_dem[valid_mask]

            if len(valid_data) == 0:
                return False, "No valid DEM data available!"

            mean_val = float(np.mean(valid_data))
            std_val = float(np.std(valid_data))

            lower_threshold = mean_val - 2 * std_val
            upper_threshold = mean_val + 2 * std_val

            processed_dem = np.copy(self.cropped_dem).astype(np.float32)

            outlier_mask_lower = valid_mask & (processed_dem < lower_threshold)
            outlier_mask_upper = valid_mask & (processed_dem > upper_threshold)

            processed_dem[outlier_mask_lower] = lower_threshold
            processed_dem[outlier_mask_upper] = upper_threshold

            self.processed_dem_2sigma = processed_dem
            self.valid_mask = valid_mask
            self.nodata_value = nodata_value

            return True, "2σ preprocessing complete"

        except Exception as e:
            return False, f"2σ preprocessing failed: {e}"

    def load_sar_mask(self, sar_path):
        """Load the SAR image and build a mask that excludes zero-value pixels"""
        try:
            with rasterio.open(sar_path) as sar_src:
                sar_data = sar_src.read(1)
                sar_transform = sar_src.transform
                sar_crs = sar_src.crs

                # Find non-zero pixels in the SAR image (valid region)
                sar_valid_mask = sar_data != 0

                # If DEM and SAR sizes differ, resample the SAR mask to the DEM size
                if sar_valid_mask.shape != self.cropped_dem.shape:
                    # Create a mask that matches the DEM dimensions
                    dem_height, dem_width = self.cropped_dem.shape

                    # Simple nearest-neighbor resampling
                    from scipy.ndimage import zoom
                    scale_y = dem_height / sar_valid_mask.shape[0]
                    scale_x = dem_width / sar_valid_mask.shape[1]

                    # Resample the SAR mask to the DEM size
                    resampled_mask = zoom(sar_valid_mask.astype(float), (scale_y, scale_x), order=0)
                    self.sar_valid_mask = (resampled_mask > 0.5).astype(bool)
                else:
                    self.sar_valid_mask = sar_valid_mask

                return True, f"SAR mask loaded successfully, valid pixel ratio: {np.sum(self.sar_valid_mask) / self.sar_valid_mask.size * 100:.1f}%"

        except Exception as e:
            # If the SAR mask fails to load, use the full region
            self.sar_valid_mask = np.ones_like(self.cropped_dem, dtype=bool)
            return False, f"Failed to load SAR mask; using the full region instead: {e}"

    def normalize_dem_to_255(self):
        try:
            # Combine the DEM valid mask with the SAR mask
            if hasattr(self, 'sar_valid_mask'):
                # Only pixels valid in both DEM and SAR participate in normalization
                combined_valid_mask = self.valid_mask & self.sar_valid_mask
                mask_info = f"(DEM valid pixels: {np.sum(self.valid_mask)}, SAR valid pixels: {np.sum(self.sar_valid_mask)}, final valid pixels: {np.sum(combined_valid_mask)})"
            else:
                combined_valid_mask = self.valid_mask
                mask_info = f"(using only DEM valid pixels: {np.sum(self.valid_mask)})"

            valid_data = self.processed_dem_2sigma[combined_valid_mask]

            if len(valid_data) == 0:
                return False, "No valid data available for normalization!"

            min_val = float(np.min(valid_data))
            max_val = float(np.max(valid_data))

            normalized_dem = np.zeros_like(self.processed_dem_2sigma, dtype=np.uint8)

            if max_val > min_val:
                normalized_values = 255 * (valid_data - min_val) / (max_val - min_val)
                normalized_dem[combined_valid_mask] = np.clip(normalized_values, 0, 255).astype(np.uint8)
            else:
                normalized_dem[combined_valid_mask] = 128

            # Set DEM regions corresponding to zero-pixel SAR areas to zero when a mask exists
            if hasattr(self, 'sar_valid_mask'):
                normalized_dem[~self.sar_valid_mask] = 0

            self.normalized_dem = normalized_dem
            return True, f"Normalization finished {mask_info}, elevation range: {min_val:.1f}m - {max_val:.1f}m"

        except Exception as e:
            return False, f"Normalization failed: {e}"

    def save_processed_dem(self, output_path):
        try:
            output_dir = os.path.dirname(output_path)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            output_profile = self.dem_profile.copy()
            output_profile.update({
                'driver': 'GTiff',
                'height': self.normalized_dem.shape[0],
                'width': self.normalized_dem.shape[1],
                'count': 1,
                'dtype': rasterio.uint8,
                'crs': self.reference_crs,
                'transform': self.cropped_transform,
                'compress': 'lzw',
                'nodata': 0,
                'tiled': True,
                'interleave': 'pixel'
            })

            with rasterio.open(output_path, 'w', **output_profile) as dst:
                dst.write(self.normalized_dem, 1)

            return True, f"Processed DEM saved: {output_path}"

        except Exception as e:
            return False, f"Failed to save: {e}"

    def process_single_sar_with_dem(self, dem_path, sar_path, output_path):
        """Process the DEM clipping aligned to a single SAR file"""
        try:
            # Load the full DEM
            success, message = self.load_dem(dem_path)
            if not success:
                return False, message

            # Load the SAR file as the reference
            success, message = self.load_reference_bounds(sar_path)
            if not success:
                return False, message

            # Crop the DEM
            success, message = self.crop_dem_to_reference()
            if not success:
                return False, message

            # Load the SAR mask (exclude zero-valued pixels)
            success, mask_message = self.load_sar_mask(sar_path)
            if success:
                print(f"  📍 {mask_message}")
            else:
                print(f"  ⚠️ {mask_message}")

            # 2σ preprocessing
            success, message = self.preprocess_dem_2sigma()
            if not success:
                return False, message

            # Normalize (now takes the SAR mask into account)
            success, message = self.normalize_dem_to_255()
            if not success:
                return False, message

            # Save
            success, message = self.save_processed_dem(output_path)
            return success, message

        except Exception as e:
            return False, f"Processing failed: {e}"


class Step4Processor:
    """Step 4 processor - DoG contour detection"""

    def __init__(self):
        # Hyper-parameters
        self.EDGE_THRESHOLD = 50
        self.THRESHOLD1 = 200
        self.KERNEL1_SIZE = 4
        self.THRESHOLD2 = 200
        self.KERNEL2_SIZE = 2
        self.THRESHOLD3 = 500
        self.ENABLE_DILATION = False
        self.DILATION_KERNEL_SIZE = 2
        self.BURR_THRESHOLD = 30
        self.SMALL_CONTOUR_THRESHOLD = 20
        self.PADDING = 5

        # Geospatial metadata
        self.geotransform = None
        self.projection = None

        # Endpoint patterns
        self.PATTERNS = np.array([
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

    def set_parameters(self, params):
        """Configure processing parameters"""
        self.EDGE_THRESHOLD = params.get('edge_threshold', 50)
        self.THRESHOLD1 = params.get('threshold1', 200)
        self.KERNEL1_SIZE = params.get('kernel1_size', 4)
        self.THRESHOLD2 = params.get('threshold2', 200)
        self.KERNEL2_SIZE = params.get('kernel2_size', 2)
        self.THRESHOLD3 = params.get('threshold3', 500)
        self.ENABLE_DILATION = params.get('enable_dilation', False)
        self.DILATION_KERNEL_SIZE = params.get('dilation_kernel_size', 2)
        self.BURR_THRESHOLD = params.get('burr_threshold', 30)
        self.SMALL_CONTOUR_THRESHOLD = params.get('small_contour_threshold', 20)
        self.PADDING = params.get('padding', 5)

    def read_geotiff_info(self, image_path):
        """Read GeoTIFF geospatial information"""
        if GDAL_AVAILABLE:
            try:
                dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
                if dataset:
                    self.geotransform = dataset.GetGeoTransform()
                    self.projection = dataset.GetProjection()
                    dataset = None
                    return True
            except Exception as e:
                pass
        return False

    def save_array_as_geotiff(self, array, output_path, datatype=gdal.GDT_Byte):
        """Save an array as a GeoTIFF with geospatial information"""
        if not GDAL_AVAILABLE or self.geotransform is None:
            if len(array.shape) == 2:
                Image.fromarray(array).save(output_path)
            else:
                Image.fromarray(array).save(output_path)
            return

        try:
            if len(array.shape) == 3:
                height, width, bands = array.shape
            else:
                height, width = array.shape
                bands = 1
                array = array[:, :, np.newaxis]

            driver = gdal.GetDriverByName('GTiff')
            dataset = driver.Create(output_path, width, height, bands, datatype)

            dataset.SetGeoTransform(self.geotransform)
            dataset.SetProjection(self.projection)

            if bands == 1:
                dataset.GetRasterBand(1).WriteArray(array[:, :, 0])
            else:
                for i in range(bands):
                    dataset.GetRasterBand(i + 1).WriteArray(array[:, :, i])

            dataset = None

        except Exception as e:
            if len(array.shape) == 2:
                Image.fromarray(array).save(output_path)
            else:
                Image.fromarray(array).save(output_path)

    def apply_dog_edge_detection(self, img_array):
        """Apply DoG edge detection"""
        g1 = ndimage.gaussian_filter(img_array, sigma=2.0)
        g2 = ndimage.gaussian_filter(img_array, sigma=6.0)
        result = g1 - g2
        result = np.absolute(result)
        result = result / result.max() * 255 if result.max() > 0 else result
        return result.astype(np.uint8)

    def filter_small_regions(self, binary_image, min_area):
        """Filter connected regions below the minimum area"""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image, connectivity=8)

        if num_labels <= 1:
            return np.zeros_like(binary_image)

        areas = stats[1:, cv2.CC_STAT_AREA]
        valid_mask = areas >= min_area

        if not np.any(valid_mask):
            return np.zeros_like(binary_image)

        valid_labels = np.where(valid_mask)[0] + 1
        filtered_mask = np.isin(labels, valid_labels).astype(np.uint8) * 255

        return filtered_mask

    def apply_zhang_suen_thinning(self, binary_image):
        """Apply the Zhang-Suen thinning algorithm"""
        binary = binary_image.copy()
        if np.max(binary) > 1:
            binary = (binary > 0).astype(np.uint8) * 255

        try:
            # Try OpenCV's thinning implementation
            thinned = cv2.ximgproc.thinning(binary)

            if self.ENABLE_DILATION:
                kernel = np.ones((self.DILATION_KERNEL_SIZE, self.DILATION_KERNEL_SIZE), np.uint8)
                thinned = cv2.dilate(thinned, kernel, iterations=1)
        except AttributeError:
            # Fall back to a morphological alternative when cv2.ximgproc.thinning is unavailable
            print("    ⚠️ cv2.ximgproc.thinning is unavailable; using morphological operations instead")
            kernel = np.ones((3, 3), np.uint8)
            thinned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        except Exception as e:
            print(f"    ⚠️ Thinning failed: {e}; using the original binary image")
            thinned = binary

        return thinned

    def generate_all_pattern_transforms(self, patterns):
        """Generate every transform of the endpoint patterns"""
        all_transforms = []

        for pattern in patterns:
            transforms_set = set()
            base_patterns = [pattern, np.fliplr(pattern)]

            for base_pattern in base_patterns:
                for rotation in range(4):
                    transformed = np.rot90(base_pattern, rotation)
                    pattern_tuple = tuple(transformed.flatten())
                    transforms_set.add(pattern_tuple)

            for pattern_tuple in transforms_set:
                all_transforms.append(np.array(pattern_tuple).reshape(5, 5))

        return np.array(all_transforms)

    def find_endpoints_vectorized(self, binary_image):
        """Vectorized endpoint detection"""
        binary = (binary_image == 255).astype(np.float32)
        h, w = binary.shape

        kernel_8 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)
        neighbor_count = cv2.filter2D(binary, -1, kernel_8)
        traditional_endpoints = (binary == 1) & (neighbor_count == 1)

        all_pattern_transforms = self.generate_all_pattern_transforms(self.PATTERNS)
        pattern_endpoints = np.zeros((h, w), dtype=bool)

        for transformed_pattern in all_pattern_transforms:
            pattern_mask = (transformed_pattern == 1)
            pattern_sum = np.sum(pattern_mask)

            if pattern_sum > 0:
                conv_result = cv2.filter2D(binary, -1, pattern_mask.astype(np.float32))
                matches = (conv_result == pattern_sum) & (binary == 1)

                matches[:2, :] = False
                matches[-2:, :] = False
                matches[:, :2] = False
                matches[:, -2:] = False

                pattern_endpoints |= matches

        traditional_coords = np.column_stack(np.where(traditional_endpoints))
        pattern_coords = np.column_stack(np.where(pattern_endpoints & ~traditional_endpoints))

        if len(traditional_coords) == 0 and len(pattern_coords) == 0:
            return []
        elif len(traditional_coords) == 0:
            return pattern_coords.tolist()
        elif len(pattern_coords) == 0:
            return traditional_coords.tolist()
        else:
            return np.vstack([traditional_coords, pattern_coords]).tolist()

    def get_8_neighbor_points(self, binary_image, point):
        """Return 8-neighborhood points whose value equals 255"""
        y, x = point
        height, width = binary_image.shape

        if binary_image[y, x] == 0:
            return 0, []

        neighbor_points = []
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        for dy, dx in offsets:
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                if binary_image[ny, nx] == 255:
                    neighbor_points.append((ny, nx))

        return len(neighbor_points), neighbor_points

    def remove_burr_fast(self, binary_image):
        """Enhanced burr-removal algorithm based on vectorized endpoint detection"""
        result_image = binary_image.copy()
        endpoints = self.find_endpoints_vectorized(result_image)

        if not endpoints:
            return result_image

        removed_burr_count = 0

        for terminal_point in endpoints:
            if result_image[terminal_point[0], terminal_point[1]] == 0:
                continue

            last_pt = terminal_point
            next_pt = terminal_point
            branch_length = 0
            flag_end = True
            burr_points = []

            while flag_end:
                neighbor_count, neighbor_points = self.get_8_neighbor_points(result_image, next_pt)

                if neighbor_count == 1:
                    burr_points.append(next_pt)
                    last_pt = next_pt
                    next_pt = neighbor_points[0]
                    branch_length += 1

                elif neighbor_count == 2:
                    burr_points.append(next_pt)
                    if last_pt != neighbor_points[0]:
                        last_pt = next_pt
                        next_pt = neighbor_points[0]
                    else:
                        last_pt = next_pt
                        next_pt = neighbor_points[1]
                    branch_length += 1

                elif neighbor_count >= 3:
                    flag_end = False

                else:
                    flag_end = False

                if branch_length > self.BURR_THRESHOLD:
                    flag_end = False

            if branch_length < self.BURR_THRESHOLD and len(burr_points) > 0:
                for burr_point in burr_points:
                    result_image[burr_point[0], burr_point[1]] = 0
                removed_burr_count += 1

        return result_image

    def remove_small_contours(self, binary_image):
        """Remove small contours"""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image, connectivity=8)

        if num_labels <= 1:
            return np.zeros_like(binary_image)

        areas = stats[1:, cv2.CC_STAT_AREA]
        valid_mask = areas >= self.SMALL_CONTOUR_THRESHOLD

        if not np.any(valid_mask):
            return np.zeros_like(binary_image)

        valid_labels = np.where(valid_mask)[0] + 1
        filtered_mask = np.isin(labels, valid_labels).astype(np.uint8) * 255

        return filtered_mask

    def calculate_contour_curvature(self, contour_points):
        """Compute the average curvature feature of a contour"""
        if len(contour_points) < 5:
            return 0.0

        try:
            # Use OpenCV to compute the convex hull
            hull = cv2.convexHull(contour_points)
            hull_area = cv2.contourArea(hull)
            contour_area = cv2.contourArea(contour_points)

            # Evaluate convexity defects via the hull area versus contour area
            if hull_area > 0:
                convexity_defect = 1 - (contour_area / hull_area)
            else:
                convexity_defect = 0

            # Compute the perimeter-to-area ratio as a shape complexity metric
            perimeter = cv2.arcLength(contour_points, True)
            if contour_area > 0:
                shape_complexity = (perimeter ** 2) / (4 * np.pi * contour_area)
            else:
                shape_complexity = 0

            # Combine convexity defect and shape complexity
            curvature_feature = convexity_defect * shape_complexity

            return round(curvature_feature, 4)
        except:
            return 0.0

    def calculate_roughness(self, contour_points):
        """Calculate contour roughness"""
        try:
            # Contour perimeter
            contour_perimeter = cv2.arcLength(contour_points, True)

            # Convex hull
            hull = cv2.convexHull(contour_points)
            hull_perimeter = cv2.arcLength(hull, True)

            # Roughness equals the contour perimeter divided by hull perimeter
            if hull_perimeter > 0:
                roughness = contour_perimeter / hull_perimeter
                return round(roughness, 4)
            else:
                return 1.0  # Return 1 (smooth) when hull perimeter is 0
        except:
            return 1.0

    def calculate_pixel_statistics(self, region_mask, original_gray_array):
        """Compute pixel statistics within the contour area"""
        try:
            # Extract pixel values inside the mask
            region_pixels = original_gray_array[region_mask]

            if len(region_pixels) > 0:
                pixel_mean = round(np.mean(region_pixels), 2)
                pixel_std = round(np.std(region_pixels), 2)
                pixel_median = round(np.median(region_pixels), 2)
                pixel_min = int(np.min(region_pixels))
                pixel_max = int(np.max(region_pixels))

                return {
                    'mean': pixel_mean,
                    'std': pixel_std,
                    'median': pixel_median,
                    'min': pixel_min,
                    'max': pixel_max
                }
            else:
                return {
                    'mean': 0.0,
                    'std': 0.0,
                    'median': 0.0,
                    'min': 0,
                    'max': 0
                }
        except:
            return {
                'mean': 0.0,
                'std': 0.0,
                'median': 0.0,
                'min': 0,
                'max': 0
            }

    def process_image(self, img_array, progress_callback=None):
        """Main image-processing workflow"""
        results = {}

        if progress_callback:
            progress_callback("Step 1: DoG edge detection")
        edges = self.apply_dog_edge_detection(img_array)
        results['edges'] = edges

        if progress_callback:
            progress_callback("Step 2: Binarization")
        binary_edges = (edges > self.EDGE_THRESHOLD).astype(np.uint8) * 255
        results['binary_edges'] = binary_edges

        if progress_callback:
            progress_callback("Step 3: Connected-component filter #1")
        filtered_mask1 = self.filter_small_regions(binary_edges, self.THRESHOLD1)
        results['filtered_mask1'] = filtered_mask1

        if progress_callback:
            progress_callback("Step 4: Opening operation")
        kernel1 = np.ones((self.KERNEL1_SIZE, self.KERNEL1_SIZE), np.uint8)
        opened_mask = cv2.morphologyEx(filtered_mask1, cv2.MORPH_OPEN, kernel1)
        results['opened_mask'] = opened_mask

        if progress_callback:
            progress_callback("Step 5: Connected-component filter #2")
        filtered_mask2 = self.filter_small_regions(opened_mask, self.THRESHOLD2)
        results['filtered_mask2'] = filtered_mask2

        if progress_callback:
            progress_callback("Step 6: Erosion")
        kernel2 = np.ones((self.KERNEL2_SIZE, self.KERNEL2_SIZE), np.uint8)
        eroded_mask = cv2.erode(filtered_mask2, kernel2, iterations=1)
        results['eroded_mask'] = eroded_mask

        if progress_callback:
            progress_callback("Step 7: Connected-component filter #3")
        filtered_mask3 = self.filter_small_regions(eroded_mask, self.THRESHOLD3)
        results['filtered_mask3'] = filtered_mask3

        if progress_callback:
            progress_callback("Step 8: Thinning")
        thinned_mask = self.apply_zhang_suen_thinning(filtered_mask3)
        results['thinned_mask'] = thinned_mask

        if progress_callback:
            progress_callback("Step 9: First burr removal")
        deburr_mask1 = self.remove_burr_fast(thinned_mask)
        results['deburr_mask1'] = deburr_mask1

        if progress_callback:
            progress_callback("Step 10: Second burr removal")
        deburr_mask2 = self.remove_burr_fast(deburr_mask1)
        results['deburr_mask2'] = deburr_mask2

        if progress_callback:
            progress_callback("Step 11: Remove small contours")
        final_mask = self.remove_small_contours(deburr_mask2)
        results['final_mask'] = final_mask

        return results

    def save_results(self, original_img_array, results, output_dir, dem_processor):
        """Save every processing artifact"""
        os.makedirs(output_dir, exist_ok=True)

        # Ensure original_img_array is a numpy array
        if not isinstance(original_img_array, np.ndarray):
            original_img_array = np.array(original_img_array)

        # Save the original image
        original_img_path = os.path.join(output_dir, "step0_original.tif")
        self.save_array_as_geotiff(original_img_array, original_img_path)

        # Persist intermediate outputs
        steps = [
            ('step2_edge_binary', results['binary_edges']),
            ('step3_filtered1', results['filtered_mask1']),
            ('step4_opened', results['opened_mask']),
            ('step5_filtered2', results['filtered_mask2']),
            ('step6_eroded', results['eroded_mask']),
            ('step7_filtered3', results['filtered_mask3']),
            ('step8_thinned', results['thinned_mask']),
            ('step9_deburr1', results['deburr_mask1']),
            ('step10_deburr2', results['deburr_mask2']),
            ('step11_final', results['final_mask'])
        ]

        for step_name, mask in steps:
            # Save the mask
            mask_path = os.path.join(output_dir, f"{step_name}_mask.tif")
            self.save_array_as_geotiff(mask, mask_path)

            # Save the result overlaid on the original image
            result_img = original_img_array.copy()

            # Ensure result_img is RGB
            if len(result_img.shape) == 2:
                # Expand grayscale to 3 channels
                result_img = np.stack([result_img, result_img, result_img], axis=-1)
            elif len(result_img.shape) == 3 and result_img.shape[-1] == 1:
                # Expand (H, W, 1) to (H, W, 3)
                result_img = np.repeat(result_img, 3, axis=-1)

            # Highlight mask pixels in red
            if len(result_img.shape) == 3 and result_img.shape[-1] >= 3:
                result_img[mask > 0] = [255, 0, 0]

            result_path = os.path.join(output_dir, f"{step_name}_marked.tif")
            self.save_array_as_geotiff(result_img, result_path)

        # Extract mask regions and compute DEM statistics
        # Convert RGB original image to grayscale for statistics
        if len(original_img_array.shape) == 3:
            original_gray_array = np.mean(original_img_array, axis=2).astype(np.uint8)
        else:
            original_gray_array = original_img_array

        self.extract_and_save_mask_regions_with_dem(
            results['final_mask'],
            original_img_array,
            original_gray_array,
            output_dir,
            dem_processor
        )

        return True

    def extract_and_save_mask_regions_with_dem(self, final_mask, original_img_array, original_gray_array, output_dir,
                                               dem_processor):
        """Enhanced routine: extract mask components, compute features, log upper-left coordinates"""
        # Connected-component analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask, connectivity=8)

        # Copy the original image for annotation
        numbered_img_with_contours = original_img_array.copy()

        # Overlay green contour lines for clarity
        numbered_img_with_contours[final_mask > 0] = [0, 255, 0]  # green contours

        print(f"Detected {num_labels - 1} connected regions")

        # Storage for statistics
        statistics_data = []
        # Storage for location metadata
        position_data = []

        # Iterate through each component (skip background index 0)
        for i in range(1, num_labels):
            print(f"  Processing component {i}/{num_labels - 1}...")

            # Bounding box statistics
            x = stats[i, cv2.CC_STAT_LEFT]  # left column
            y = stats[i, cv2.CC_STAT_TOP]  # top row
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            # Record PNG index and coordinate
            position_data.append([
                i,  # PNG index
                x,  # left column
                y  # top row
            ])

            # Use the longer side of the bounding box
            max_dimension = max(w, h)

            # Center of the bounding box
            center_x = x + w // 2
            center_y = y + h // 2

            # Build a square window padded by PADDING
            half_size = (max_dimension + 2 * self.PADDING) // 2

            height, width = original_img_array.shape[:2]
            square_x_min = max(0, center_x - half_size)
            square_y_min = max(0, center_y - half_size)
            square_x_max = min(width, center_x + half_size)
            square_y_max = min(height, center_y + half_size)

            # Enforce a square even if clipping at edges
            actual_width = square_x_max - square_x_min
            actual_height = square_y_max - square_y_min
            actual_size = min(actual_width, actual_height)

            # Recompute the square bounds
            square_x_max = square_x_min + actual_size
            square_y_max = square_y_min + actual_size

            # Memory-friendly: operate within the cropped window
            try:
                # Local window labels
                cropped_labels = labels[square_y_min:square_y_max, square_x_min:square_x_max]

                # Mask for the current component
                cropped_mask = (cropped_labels == i)

                # Count contour pixels
                edge_pixel_count = np.sum(cropped_mask)

                # Aspect ratio metric: width/height + height/width
                # Guard division-by-zero
                if w > 0 and h > 0:
                    ratio_sum = (w / h) + (h / w)
                else:
                    ratio_sum = 0

                # === Geometric features ===
                # Full mask of the component
                region_full_mask = (labels == i)

                # Find contour points
                contours, _ = cv2.findContours(
                    region_full_mask.astype(np.uint8),
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                if len(contours) > 0:
                    # Use the largest contour
                    main_contour = max(contours, key=cv2.contourArea)

                    # Evaluate geometric features
                    curvature_feature = self.calculate_contour_curvature(main_contour)
                    roughness = self.calculate_roughness(main_contour)
                else:
                    curvature_feature = 0.0
                    roughness = 1.0  # default to smooth

                # === Pixel statistics (contour pixels only) ===
                region_contour_mask = (labels == i) & (final_mask > 0)
                pixel_stats = self.calculate_pixel_statistics(region_contour_mask, original_gray_array)

                # === DEM statistics ===
                if dem_processor:
                    dem_stats = dem_processor.calculate_dem_statistics_for_region(region_contour_mask)
                else:
                    dem_stats = {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}

                # Create a square grayscale patch
                gray_bg = np.zeros((actual_size, actual_size), dtype=np.uint8)

                # Fill with white (255) where the region exists
                gray_bg[cropped_mask] = 255

                # Annotate the original image with IDs
                centroid_x = int(centroids[i][0])
                centroid_y = int(centroids[i][1])

                # Use a readable font size and bright color
                font_scale = 1.2
                thickness = 3

                # Overlay red text directly
                text = str(i)
                (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                                                      thickness)

                # Draw the label
                cv2.putText(numbered_img_with_contours, text,
                            (centroid_x - text_width // 2, centroid_y),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), thickness)

                # Save the square patch as grayscale PNG
                output_path = os.path.join(output_dir, f"{i}.png")
                Image.fromarray(gray_bg, mode='L').save(output_path)

                # Collect statistics (enhanced version + DEM info)
                statistics_data.append([
                    i,
                    edge_pixel_count,
                    w,
                    h,
                    round(ratio_sum, 3),
                    curvature_feature,
                    roughness,
                    pixel_stats['mean'],
                    pixel_stats['std'],
                    pixel_stats['median'],
                    dem_stats['mean'],
                    dem_stats['std'],
                    dem_stats['median'],
                    dem_stats['min'],
                    dem_stats['max'],
                    dem_stats['count'],
                    ""
                ])

                # Release memory
                del cropped_labels, cropped_mask, gray_bg, region_full_mask, region_contour_mask
                gc.collect()

            except MemoryError:
                print(f"    Warning: component {i} is too large; skip saving. Region size: {actual_size}x{actual_size}")
                continue
            except Exception as e:
                print(f"    Warning: failed while processing component {i}: {e}")
                continue

        # Persist the statistics
        if statistics_data:
            # Prefer Excel via pandas
            try:
                import pandas as pd
                xlsx_path = os.path.join(output_dir, "contour_statistics_enhanced_with_dem.xlsx")

                df_stats = pd.DataFrame(statistics_data, columns=[
                    'PNG Index',
                    'Edge Pixels',
                    'Bounding Box Width',
                    'Bounding Box Height',
                    'Width/Height + Height/Width',
                    'Curvature Feature',
                    'Roughness',
                    'Pixel Mean',
                    'Pixel Std',
                    'Pixel Median',
                    'DEM Mean',
                    'DEM Std',
                    'DEM Median',
                    'DEM Min',
                    'DEM Max',
                    'DEM Valid Pixel Count',
                    'Category'
                ])

                df_stats.to_excel(xlsx_path, index=False, engine='openpyxl')
                print(f"  ✅ Generated enhanced DEM-aware Excel stats: {xlsx_path}")
            except ImportError:
                csv_path = os.path.join(output_dir, "contour_statistics_enhanced_with_dem.csv")
                with open(csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([
                        'PNG Index', 'Edge Pixels', 'Bounding Box Width', 'Bounding Box Height',
                        'Width/Height + Height/Width', 'Curvature Feature', 'Roughness',
                        'Pixel Mean', 'Pixel Std', 'Pixel Median',
                        'DEM Mean', 'DEM Std', 'DEM Median', 'DEM Min', 'DEM Max',
                        'DEM Valid Pixel Count', 'Category'
                    ])
                    writer.writerows(statistics_data)
                print(f"  ✅ Generated DEM-aware CSV stats file: {csv_path}")

        # Save position metadata
        if position_data:
            try:
                import pandas as pd
                position_xlsx_path = os.path.join(output_dir, "contour_positions.xlsx")

                df_positions = pd.DataFrame(position_data, columns=[
                    'PNG Index',
                    'Top-left X',
                    'Top-left Y'
                ])

                df_positions.to_excel(position_xlsx_path, index=False, engine='openpyxl')
                print(f"  ✅ Generated contour position Excel file: {position_xlsx_path}")
            except ImportError:
                position_csv_path = os.path.join(output_dir, "contour_positions.csv")
                with open(position_csv_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['PNG Index', 'Top-left X', 'Top-left Y'])
                    writer.writerows(position_data)
                print(f"  ✅ Generated contour position CSV file: {position_csv_path}")

        # Save labeled contour TIFF
        numbered_contours_path = os.path.join(output_dir, "numbered_regions_with_contours_enhanced_dem.tif")
        self.save_array_as_geotiff(numbered_img_with_contours, numbered_contours_path)
        print(f"  ✅ Saved DEM-aware annotated contour image to {numbered_contours_path}")

        # Save a contours-only version for quick inspection
        contours_only_img = original_img_array.copy()
        contours_only_img[final_mask > 0] = [255, 0, 0]  # red contours
        contours_only_path = os.path.join(output_dir, "contours_only_enhanced_dem.tif")
        self.save_array_as_geotiff(contours_only_img, contours_only_path)
        print(f"  ✅ Saved DEM-aware contour-only image to {contours_only_path}")

        print(f"  📊 Compiled statistics for {len(statistics_data)} connected regions")
        return numbered_img_with_contours


class DEMProcessor:
    """DEM data processor"""

    def __init__(self, dem_path, sar_path):
        self.dem_data = None
        self.dem_transform = None
        self.dem_bounds = None
        self.sar_transform = None
        self.sar_bounds = None
        self.coord_cache = {}

        self.load_dem_data(dem_path)
        self.load_sar_data(sar_path)
        self.prepare_coordinate_mapping()

    def load_dem_data(self, dem_path):
        try:
            with rasterio.open(dem_path) as dem_src:
                self.dem_data = dem_src.read(1)
                self.dem_transform = dem_src.transform
                self.dem_bounds = dem_src.bounds
                self.dem_crs = dem_src.crs
                self.dem_nodata = dem_src.nodata
                return True
        except Exception as e:
            return False

    def load_sar_data(self, sar_path):
        try:
            with rasterio.open(sar_path) as sar_src:
                self.sar_transform = sar_src.transform
                self.sar_bounds = sar_src.bounds
                self.sar_crs = sar_src.crs
                self.sar_shape = (sar_src.height, sar_src.width)
                return True
        except Exception as e:
            return False

    def prepare_coordinate_mapping(self):
        if self.dem_crs != self.sar_crs:
            pass

        overlap_left = max(self.dem_bounds.left, self.sar_bounds.left)
        overlap_right = min(self.dem_bounds.right, self.sar_bounds.right)
        overlap_bottom = max(self.dem_bounds.bottom, self.sar_bounds.bottom)
        overlap_top = min(self.dem_bounds.top, self.sar_bounds.top)

        self.overlap_bounds = {
            'left': overlap_left, 'right': overlap_right,
            'bottom': overlap_bottom, 'top': overlap_top
        }

    def sar_pixel_to_dem_region(self, sar_pixels):
        if len(sar_pixels) == 0:
            return []

        sar_pixels = np.array(sar_pixels)
        sar_cols = sar_pixels[:, 1]
        sar_rows = sar_pixels[:, 0]

        geo_x = self.sar_transform[2] + sar_cols * self.sar_transform[0] + sar_rows * self.sar_transform[1]
        geo_y = self.sar_transform[5] + sar_cols * self.sar_transform[3] + sar_rows * self.sar_transform[4]

        dem_cols = (geo_x - self.dem_transform[2]) / self.dem_transform[0]
        dem_rows = (geo_y - self.dem_transform[5]) / self.dem_transform[4]

        dem_cols = np.clip(np.round(dem_cols).astype(int), 0, self.dem_data.shape[1] - 1)
        dem_rows = np.clip(np.round(dem_rows).astype(int), 0, self.dem_data.shape[0] - 1)

        return list(zip(dem_rows, dem_cols))

    def calculate_dem_statistics_for_region(self, region_mask):
        try:
            sar_pixels = np.where(region_mask)
            sar_coords = list(zip(sar_pixels[0], sar_pixels[1]))

            if len(sar_coords) == 0:
                return {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'count': 0, 'min': 0.0, 'max': 0.0}

            dem_coords = self.sar_pixel_to_dem_region(sar_coords)

            dem_values = []
            for dem_row, dem_col in dem_coords:
                try:
                    dem_value = self.dem_data[dem_row, dem_col]
                    if self.dem_nodata is None or dem_value != self.dem_nodata:
                        if dem_value != 0:
                            dem_values.append(dem_value)
                except (IndexError, TypeError):
                    continue

            if len(dem_values) > 0:
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
            return {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}


class WorkerThread(QThread):
    """Background worker thread"""
    progress_updated = pyqtSignal(int)
    status_updated = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, task_type, **kwargs):
        super().__init__()
        self.task_type = task_type
        self.kwargs = kwargs

    def run(self):
        try:
            if self.task_type == "step1_single":
                self._run_step1_single()
            elif self.task_type == "step1_batch":
                self._run_step1_batch()
            elif self.task_type == "step2":
                self._run_step2()
            elif self.task_type == "step3_single":
                self._run_step3_single()
            elif self.task_type == "step3_batch":
                self._run_step3_batch()
            elif self.task_type == "step4_single":
                self._run_step4_single()
            elif self.task_type == "step4_batch":
                self._run_step4_batch()
            elif self.task_type == "step4":  # Backward compatibility
                self._run_step4_single()
        except Exception as e:
            self.finished_signal.emit(False, f"Processing error: {str(e)}")

    def _run_step1_single(self):
        input_path = self.kwargs['input_path']
        output_path = self.kwargs['output_path']

        self.status_updated.emit("Processing single file...")
        self.progress_updated.emit(20)

        success, message = Step1Processor.process_single_tif(input_path, output_path)

        self.progress_updated.emit(100)
        self.finished_signal.emit(success, message)

    def _run_step1_batch(self):
        input_folder = self.kwargs['input_folder']
        output_folder = self.kwargs['output_folder']

        tif_files = []
        for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
            tif_files.extend(glob.glob(os.path.join(input_folder, ext)))

        if not tif_files:
            self.finished_signal.emit(False, "No TIF files found in the input folder!")
            return

        total_files = len(tif_files)
        success_count = 0

        for i, tif_file in enumerate(tif_files):
            filename = os.path.basename(tif_file)
            name_without_ext = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, f"{name_without_ext}_processed.tif")

            self.status_updated.emit(f"Processing ({i + 1}/{total_files}): {filename}")

            success, message = Step1Processor.process_single_tif(tif_file, output_path)
            if success:
                success_count += 1

            progress = int((i + 1) / total_files * 100)
            self.progress_updated.emit(progress)

        result_message = f"Batch processing complete! {success_count}/{total_files} files succeeded"
        self.finished_signal.emit(True, result_message)

    def _run_step2(self):
        input_path = self.kwargs['input_path']
        output_path = self.kwargs['output_path']
        delta_lon = self.kwargs['delta_lon']
        delta_lat = self.kwargs['delta_lat']

        processor = Step2Processor()

        self.status_updated.emit("Loading SAR image...")
        self.progress_updated.emit(20)

        success, message = processor.load_sar_image(input_path)
        if not success:
            self.finished_signal.emit(False, message)
            return

        self.status_updated.emit("Adjusting geolocation...")
        self.progress_updated.emit(50)

        success, message = processor.adjust_geolocation(delta_lon, delta_lat)
        if not success:
            self.finished_signal.emit(False, message)
            return

        self.status_updated.emit("Saving adjusted image...")
        self.progress_updated.emit(80)

        success, message = processor.save_adjusted_sar(output_path)

        self.progress_updated.emit(100)
        self.finished_signal.emit(success, message)

    def _run_step3_single(self):
        """Run Step 3 for a single file"""
        dem_path = self.kwargs['dem_path']
        reference_path = self.kwargs['reference_path']
        output_path = self.kwargs['output_path']

        processor = Step3Processor()

        self.status_updated.emit("Loading reference SAR file...")
        self.progress_updated.emit(10)

        success, message = processor.load_reference_bounds(reference_path)
        if not success:
            self.finished_signal.emit(False, message)
            return

        self.status_updated.emit("Loading DEM file...")
        self.progress_updated.emit(20)

        success, message = processor.load_dem(dem_path)
        if not success:
            self.finished_signal.emit(False, message)
            return

        self.status_updated.emit("Cropping DEM...")
        self.progress_updated.emit(40)

        success, message = processor.crop_dem_to_reference()
        if not success:
            self.finished_signal.emit(False, message)
            return

        self.status_updated.emit("Loading SAR mask (excludes zero-value pixels)...")
        self.progress_updated.emit(50)

        success, mask_message = processor.load_sar_mask(reference_path)
        if success:
            self.status_updated.emit(f"✅ {mask_message}")
        else:
            self.status_updated.emit(f"⚠️ {mask_message}")

        self.status_updated.emit("2σ preprocessing...")
        self.progress_updated.emit(60)

        success, message = processor.preprocess_dem_2sigma()
        if not success:
            self.finished_signal.emit(False, message)
            return

        self.status_updated.emit("Normalizing (applying SAR mask)...")
        self.progress_updated.emit(80)

        success, message = processor.normalize_dem_to_255()
        if not success:
            self.finished_signal.emit(False, message)
            return

        self.status_updated.emit("Saving processed DEM...")
        self.progress_updated.emit(90)

        success, message = processor.save_processed_dem(output_path)

        self.progress_updated.emit(100)
        self.finished_signal.emit(success, message)

    def _run_step3_batch(self):
        """Run Step 3 in batch mode"""
        dem_path = self.kwargs['dem_path']
        sar_folder = self.kwargs['sar_folder']
        output_folder = self.kwargs['output_folder']

        # Gather SAR files
        sar_files = []
        for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
            sar_files.extend(glob.glob(os.path.join(sar_folder, ext)))

        if not sar_files:
            self.finished_signal.emit(False, "No TIF files found in the SAR folder!")
            return

        total_files = len(sar_files)
        successful_count = 0

        self.status_updated.emit(f"Starting batch run for {total_files} SAR files...")

        # Ensure the output directory exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        for i, sar_file in enumerate(sar_files):
            try:
                # Base filename without extension
                sar_basename = os.path.splitext(os.path.basename(sar_file))[0]

                # Output DEM path
                output_dem_path = os.path.join(output_folder, f"{sar_basename}_dem_processed.tif")

                self.status_updated.emit(f"Processing ({i + 1}/{total_files}): {sar_basename}")

                # Use a fresh processor to avoid state contamination
                processor = Step3Processor()

                # Process DEM based on SAR footprint
                success, message = processor.process_single_sar_with_dem(dem_path, sar_file, output_dem_path)

                if success:
                    successful_count += 1
                    self.status_updated.emit(f"  ✅ Done: {sar_basename}")
                else:
                    self.status_updated.emit(f"  ❌ Failed: {sar_basename} - {message}")

                # Progress update
                progress = int((i + 1) / total_files * 100)
                self.progress_updated.emit(progress)

                # Free memory
                del processor
                gc.collect()

            except Exception as e:
                self.status_updated.emit(f"  ❌ Error: {os.path.basename(sar_file)} - {e}")
                continue

        # Wrap up
        self.progress_updated.emit(100)
        result_message = f"Batch DEM processing finished! {successful_count}/{total_files} succeeded"
        self.status_updated.emit(result_message)
        self.finished_signal.emit(True, result_message)

    def _run_step4_single(self):
        """Run Step 4 for a single input"""
        sar_path = self.kwargs['sar_path']
        dem_path = self.kwargs['dem_path']
        output_dir = self.kwargs['output_dir']
        parameters = self.kwargs['parameters']

        processor = Step4Processor()
        processor.set_parameters(parameters)

        self.status_updated.emit("Reading geospatial metadata...")
        self.progress_updated.emit(5)

        processor.read_geotiff_info(sar_path)

        self.status_updated.emit("Loading large image via rasterio...")
        self.progress_updated.emit(10)

        try:
            # Use rasterio for large images to avoid PIL limits
            with rasterio.open(sar_path) as src:
                # Read the first band
                img_array = src.read(1)
                img_profile = src.profile

                # Normalize to uint8 when needed
                if img_array.dtype != np.uint8:
                    # Normalize floats to 0-255
                    if img_array.dtype in [np.float32, np.float64]:
                        img_min, img_max = np.min(img_array), np.max(img_array)
                        if img_max > img_min:
                            img_array = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                        else:
                            img_array = np.zeros_like(img_array, dtype=np.uint8)
                    else:
                        # Clip/scale other integer types
                        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

                # Build RGB visualization (stack grayscale)
                height, width = img_array.shape
                original_img_array = np.stack([img_array, img_array, img_array], axis=-1)

                self.status_updated.emit(f"✅ Image loaded: {width}x{height} pixels ({width * height:,} total)")

        except Exception as e:
            self.finished_signal.emit(False, f"Failed to read image with rasterio: {e}")
            return

        self.status_updated.emit("Initializing DEM processor...")
        self.progress_updated.emit(15)

        try:
            dem_processor = DEMProcessor(dem_path, sar_path)
        except Exception as e:
            self.finished_signal.emit(False, f"Failed to initialize DEM processor: {e}")
            return

        self.status_updated.emit("Starting image processing...")
        self.progress_updated.emit(20)

        # Process the image
        def progress_callback(message):
            self.status_updated.emit(message)

        try:
            results = processor.process_image(img_array, progress_callback)
        except Exception as e:
            self.finished_signal.emit(False, f"Image processing failed: {e}")
            return

        self.status_updated.emit("Saving processing artifacts...")
        self.progress_updated.emit(90)

        try:
            # Pass numpy arrays directly to save_results
            processor.save_results(original_img_array, results, output_dir, dem_processor)
        except Exception as e:
            self.finished_signal.emit(False, f"Failed to save outputs: {e}")
            return

        self.progress_updated.emit(100)
        self.finished_signal.emit(True, f"Contour detection finished! Results saved to: {output_dir}")

    def _run_step4_batch(self):
        """Run Step 4 in batch mode"""
        sar_folder = self.kwargs['sar_folder']
        dem_path_or_folder = self.kwargs['dem_path_or_folder']
        output_dir = self.kwargs['output_dir']
        dem_mode = self.kwargs['dem_mode']
        parameters = self.kwargs['parameters']

        # Gather SAR files
        sar_files = []
        for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
            sar_files.extend(glob.glob(os.path.join(sar_folder, ext)))

        if not sar_files:
            self.finished_signal.emit(False, "No TIF files found in the SAR folder!")
            return

        total_files = len(sar_files)
        successful_count = 0

        self.status_updated.emit(f"Starting batch contour detection for {total_files} files...")

        for i, sar_file in enumerate(sar_files):
            try:
                # Derive base name
                sar_basename = os.path.splitext(os.path.basename(sar_file))[0]

                # Create per-file output directory
                file_output_dir = os.path.join(output_dir, sar_basename)
                os.makedirs(file_output_dir, exist_ok=True)

                self.status_updated.emit(f"Processing file ({i + 1}/{total_files}): {sar_basename}")

                # Determine DEM file
                if dem_mode == "Single DEM File":
                    dem_file = dem_path_or_folder
                else:  # Match per file
                    # Search for a matching DEM file
                    possible_dem_files = []
                    for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
                        possible_dem_files.extend(glob.glob(os.path.join(dem_path_or_folder, ext)))

                    # Try to find the best match (same or overlapping name)
                    dem_file = None
                    for dem_candidate in possible_dem_files:
                        dem_basename = os.path.splitext(os.path.basename(dem_candidate))[0]
                        if dem_basename == sar_basename or sar_basename in dem_basename or dem_basename in sar_basename:
                            dem_file = dem_candidate
                            break

                    if dem_file is None:
                        # Fall back to the first DEM when no match exists
                        if possible_dem_files:
                            dem_file = possible_dem_files[0]
                            self.status_updated.emit(f"  Warning: DEM match not found; using {os.path.basename(dem_file)}")
                        else:
                            self.status_updated.emit(f"  Error: DEM folder contains no TIF files; skipping {sar_basename}")
                            continue

                # Validate DEM exists
                if not os.path.exists(dem_file):
                    self.status_updated.emit(f"  Error: DEM file missing; skipping {sar_basename}")
                    continue

                # Configure processor
                processor = Step4Processor()
                processor.set_parameters(parameters)

                # Load geospatial metadata
                processor.read_geotiff_info(sar_file)

                # Load SAR image
                try:
                    # Use rasterio for large images
                    with rasterio.open(sar_file) as src:
                        # Read first band
                        img_array = src.read(1)
                        img_profile = src.profile

                        # Normalize dtype
                        if img_array.dtype != np.uint8:
                            # Normalize floats
                            if img_array.dtype in [np.float32, np.float64]:
                                img_min, img_max = np.min(img_array), np.max(img_array)
                                if img_max > img_min:
                                    img_array = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                                else:
                                    img_array = np.zeros_like(img_array, dtype=np.uint8)
                            else:
                                # Clip integers
                                img_array = np.clip(img_array, 0, 255).astype(np.uint8)

                        # Create RGB stack
                        height, width = img_array.shape
                        original_img_array = np.stack([img_array, img_array, img_array], axis=-1)

                        self.status_updated.emit(f"  ✅ rasterio read: {width}x{height} ({width * height:,} pixels)")

                except Exception as e:
                    self.status_updated.emit(f"  Error: rasterio failed for {sar_basename}: {e}")
                    continue

                # Initialize DEM processor
                try:
                    dem_processor = DEMProcessor(dem_file, sar_file)
                except Exception as e:
                    self.status_updated.emit(f"  Error: DEM processor init failed for {sar_basename}: {e}")
                    continue

                # Process image
                def progress_callback(message):
                    self.status_updated.emit(f"  {sar_basename}: {message}")

                try:
                    results = processor.process_image(img_array, progress_callback)
                except Exception as e:
                    self.status_updated.emit(f"  Error: processing failed for {sar_basename}: {e}")
                    continue

                # Save outputs
                try:
                    processor.save_results(original_img_array, results, file_output_dir, dem_processor)
                    successful_count += 1
                    self.status_updated.emit(f"  ✅ Completed: {sar_basename}")
                except Exception as e:
                    self.status_updated.emit(f"  Error: failed to save outputs for {sar_basename}: {e}")
                    continue

                # Update progress
                progress = int((i + 1) / total_files * 100)
                self.progress_updated.emit(progress)

            except Exception as e:
                self.status_updated.emit(f"  Error: unexpected failure while processing {os.path.basename(sar_file)}: {e}")
                continue

        # Wrap up
        self.progress_updated.emit(100)
        result_message = f"Batch contour detection finished! {successful_count}/{total_files} successes"
        self.status_updated.emit(result_message)
        self.finished_signal.emit(True, result_message)


class Step1Widget(QWidget):
    """Step 1 widget"""

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.worker_thread = None

    def init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("Step 1: SAR Image Preprocessing")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)

        single_group = QGroupBox("Single File Processing")
        single_layout = QGridLayout()

        single_layout.addWidget(QLabel("Input TIF file:"), 0, 0)
        self.single_input_edit = QLineEdit()
        single_layout.addWidget(self.single_input_edit, 0, 1)
        single_input_btn = QPushButton("Browse")
        single_input_btn.clicked.connect(self.select_single_input)
        single_layout.addWidget(single_input_btn, 0, 2)

        single_layout.addWidget(QLabel("Output file:"), 1, 0)
        self.single_output_edit = QLineEdit()
        single_layout.addWidget(self.single_output_edit, 1, 1)
        single_output_btn = QPushButton("Browse")
        single_output_btn.clicked.connect(self.select_single_output)
        single_layout.addWidget(single_output_btn, 1, 2)

        self.single_process_btn = QPushButton("Start Single-File Run")
        self.single_process_btn.clicked.connect(self.start_single_processing)
        single_layout.addWidget(self.single_process_btn, 2, 0, 1, 3)

        single_group.setLayout(single_layout)
        layout.addWidget(single_group)

        batch_group = QGroupBox("Batch Processing")
        batch_layout = QGridLayout()

        batch_layout.addWidget(QLabel("Input folder:"), 0, 0)
        self.batch_input_edit = QLineEdit()
        batch_layout.addWidget(self.batch_input_edit, 0, 1)
        batch_input_btn = QPushButton("Browse")
        batch_input_btn.clicked.connect(self.select_batch_input)
        batch_layout.addWidget(batch_input_btn, 0, 2)

        batch_layout.addWidget(QLabel("Output folder:"), 1, 0)
        self.batch_output_edit = QLineEdit()
        batch_layout.addWidget(self.batch_output_edit, 1, 1)
        batch_output_btn = QPushButton("Browse")
        batch_output_btn.clicked.connect(self.select_batch_output)
        batch_layout.addWidget(batch_output_btn, 1, 2)

        self.batch_process_btn = QPushButton("Start Batch Run")
        self.batch_process_btn.clicked.connect(self.start_batch_processing)
        batch_layout.addWidget(self.batch_process_btn, 2, 0, 1, 3)

        batch_group.setLayout(batch_layout)
        layout.addWidget(batch_group)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(150)
        layout.addWidget(self.status_text)

        self.setLayout(layout)

    def select_single_input(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select TIF Input", "", "TIF Files (*.tif *.tiff)")
        if file_path:
            self.single_input_edit.setText(file_path)

    def select_single_output(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Output File", "", "TIF Files (*.tif)")
        if file_path:
            self.single_output_edit.setText(file_path)

    def select_batch_input(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder_path:
            self.batch_input_edit.setText(folder_path)

    def select_batch_output(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder_path:
            self.batch_output_edit.setText(folder_path)

    def start_single_processing(self):
        input_path = self.single_input_edit.text().strip()
        output_path = self.single_output_edit.text().strip()

        if not input_path or not output_path:
            QMessageBox.warning(self, "Warning", "Select both input and output paths.")
            return

        if not os.path.exists(input_path):
            QMessageBox.warning(self, "Warning", "Input file does not exist.")
            return

        self.start_processing("step1_single", input_path=input_path, output_path=output_path)

    def start_batch_processing(self):
        input_folder = self.batch_input_edit.text().strip()
        output_folder = self.batch_output_edit.text().strip()

        if not input_folder or not output_folder:
            QMessageBox.warning(self, "Warning", "Select both input and output folders.")
            return

        if not os.path.exists(input_folder):
            QMessageBox.warning(self, "Warning", "Input folder does not exist.")
            return

        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        self.start_processing("step1_batch", input_folder=input_folder, output_folder=output_folder)

    def start_processing(self, task_type, **kwargs):
        self.single_process_btn.setEnabled(False)
        self.batch_process_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_text.clear()

        self.worker_thread = WorkerThread(task_type, **kwargs)
        self.worker_thread.progress_updated.connect(self.progress_bar.setValue)
        self.worker_thread.status_updated.connect(self.append_status)
        self.worker_thread.finished_signal.connect(self.on_processing_finished)
        self.worker_thread.start()

    def append_status(self, message):
        self.status_text.append(message)

    def on_processing_finished(self, success, message):
        self.single_process_btn.setEnabled(True)
        self.batch_process_btn.setEnabled(True)

        if success:
            self.append_status(f"✅ {message}")
            QMessageBox.information(self, "Success", message)
        else:
            self.append_status(f"❌ {message}")
            QMessageBox.critical(self, "Error", message)


class Step2Widget(QWidget):
    """Step 2 widget"""

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.worker_thread = None

    def init_ui(self):
        layout = QVBoxLayout()

        title = QLabel("Step 2: SAR Geolocation Adjustment")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)

        file_group = QGroupBox("File Selection")
        file_layout = QGridLayout()

        file_layout.addWidget(QLabel("Input SAR file:"), 0, 0)
        self.input_edit = QLineEdit()
        file_layout.addWidget(self.input_edit, 0, 1)
        input_btn = QPushButton("Browse")
        input_btn.clicked.connect(self.select_input)
        file_layout.addWidget(input_btn, 0, 2)

        file_layout.addWidget(QLabel("Output file:"), 1, 0)
        self.output_edit = QLineEdit()
        file_layout.addWidget(self.output_edit, 1, 1)
        output_btn = QPushButton("Browse")
        output_btn.clicked.connect(self.select_output)
        file_layout.addWidget(output_btn, 1, 2)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        param_group = QGroupBox("Adjustment Parameters")
        param_layout = QGridLayout()

        param_layout.addWidget(QLabel("Longitude offset (degrees):"), 0, 0)
        self.lon_spinbox = QDoubleSpinBox()
        self.lon_spinbox.setRange(-1.0, 1.0)
        self.lon_spinbox.setDecimals(8)
        self.lon_spinbox.setSingleStep(0.0001)
        self.lon_spinbox.setValue(-0.073)
        param_layout.addWidget(self.lon_spinbox, 0, 1)
        param_layout.addWidget(QLabel("(positive = east, negative = west)"), 0, 2)

        param_layout.addWidget(QLabel("Latitude offset (degrees):"), 1, 0)
        self.lat_spinbox = QDoubleSpinBox()
        self.lat_spinbox.setRange(-1.0, 1.0)
        self.lat_spinbox.setDecimals(8)
        self.lat_spinbox.setSingleStep(0.0001)
        self.lat_spinbox.setValue(0.012)
        param_layout.addWidget(self.lat_spinbox, 1, 1)
        param_layout.addWidget(QLabel("(positive = north, negative = south)"), 1, 2)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        self.process_btn = QPushButton("Start Geolocation Adjustment")
        self.process_btn.clicked.connect(self.start_processing)
        layout.addWidget(self.process_btn)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(150)
        layout.addWidget(self.status_text)

        self.setLayout(layout)

    def select_input(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select SAR Input", "", "TIF Files (*.tif *.tiff)")
        if file_path:
            self.input_edit.setText(file_path)

    def select_output(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Output File", "", "TIF Files (*.tif)")
        if file_path:
            self.output_edit.setText(file_path)

    def start_processing(self):
        input_path = self.input_edit.text().strip()
        output_path = self.output_edit.text().strip()
        delta_lon = self.lon_spinbox.value()
        delta_lat = self.lat_spinbox.value()

        if not input_path or not output_path:
            QMessageBox.warning(self, "Warning", "Select both input and output files.")
            return

        if not os.path.exists(input_path):
            QMessageBox.warning(self, "Warning", "Input file does not exist.")
            return

        self.process_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_text.clear()

        self.worker_thread = WorkerThread("step2",
                                          input_path=input_path,
                                          output_path=output_path,
                                          delta_lon=delta_lon,
                                          delta_lat=delta_lat)
        self.worker_thread.progress_updated.connect(self.progress_bar.setValue)
        self.worker_thread.status_updated.connect(self.append_status)
        self.worker_thread.finished_signal.connect(self.on_processing_finished)
        self.worker_thread.start()

    def append_status(self, message):
        self.status_text.append(message)

    def on_processing_finished(self, success, message):
        self.process_btn.setEnabled(True)

        if success:
            self.append_status(f"✅ {message}")
            QMessageBox.information(self, "Success", message)
        else:
            self.append_status(f"❌ {message}")
            QMessageBox.critical(self, "Error", message)


class Step3Widget(QWidget):
    """Step 3 widget - supports single-file and batch processing"""

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.worker_thread = None

    def init_ui(self):
        layout = QVBoxLayout()

        # Title
        title = QLabel("Step 3: DEM Preprocessing")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)

        # Mode selection
        mode_group = QGroupBox("Processing Mode")
        mode_layout = QHBoxLayout()

        # Use radio buttons with a button group for exclusivity
        self.mode_button_group = QButtonGroup()

        self.single_mode_radio = QRadioButton("Single File")
        self.single_mode_radio.setChecked(True)
        self.mode_button_group.addButton(self.single_mode_radio, 0)
        mode_layout.addWidget(self.single_mode_radio)

        self.batch_mode_radio = QRadioButton("Batch")
        self.mode_button_group.addButton(self.batch_mode_radio, 1)
        mode_layout.addWidget(self.batch_mode_radio)

        # Wire up mode switch handling
        self.mode_button_group.buttonClicked.connect(self.toggle_mode)

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Single-file group
        self.single_file_group = QGroupBox("Single File")
        single_layout = QGridLayout()

        single_layout.addWidget(QLabel("DEM file:"), 0, 0)
        self.dem_edit = QLineEdit()
        single_layout.addWidget(self.dem_edit, 0, 1)
        dem_btn = QPushButton("Browse")
        dem_btn.clicked.connect(self.select_dem)
        single_layout.addWidget(dem_btn, 0, 2)

        single_layout.addWidget(QLabel("Reference SAR file:"), 1, 0)
        self.reference_edit = QLineEdit()
        single_layout.addWidget(self.reference_edit, 1, 1)
        reference_btn = QPushButton("Browse")
        reference_btn.clicked.connect(self.select_reference)
        single_layout.addWidget(reference_btn, 1, 2)

        # Guidance text
        single_info = QLabel("💡 The reference SAR image defines the cropping area; zero-value SAR pixels are excluded during DEM normalization.")
        single_info.setStyleSheet("color: #666; font-size: 11px; margin: 5px;")
        single_layout.addWidget(single_info, 3, 0, 1, 3)

        single_layout.addWidget(QLabel("Output file:"), 2, 0)
        self.output_edit = QLineEdit()
        single_layout.addWidget(self.output_edit, 2, 1)
        output_btn = QPushButton("Browse")
        output_btn.clicked.connect(self.select_output)
        single_layout.addWidget(output_btn, 2, 2)

        # Additional details
        single_info_step4 = QLabel(
            "📊 DoG contour detection produces step masks, connected-region PNG tiles, DEM statistics (Excel/CSV), and contour visualizations.")
        single_info_step4.setStyleSheet("color: #666; font-size: 11px; margin: 5px;")
        single_layout.addWidget(single_info_step4, 3, 0, 1, 3)

        self.single_file_group.setLayout(single_layout)
        layout.addWidget(self.single_file_group)

        # Batch group
        self.batch_file_group = QGroupBox("Batch")
        batch_layout = QGridLayout()

        batch_layout.addWidget(QLabel("Large DEM file:"), 0, 0)
        self.batch_dem_edit = QLineEdit()
        batch_layout.addWidget(self.batch_dem_edit, 0, 1)
        batch_dem_btn = QPushButton("Browse")
        batch_dem_btn.clicked.connect(self.select_batch_dem)
        batch_layout.addWidget(batch_dem_btn, 0, 2)

        batch_layout.addWidget(QLabel("SAR folder:"), 1, 0)
        self.sar_folder_edit = QLineEdit()
        batch_layout.addWidget(self.sar_folder_edit, 1, 1)
        sar_folder_btn = QPushButton("Browse")
        sar_folder_btn.clicked.connect(self.select_sar_folder)
        batch_layout.addWidget(sar_folder_btn, 1, 2)

        batch_layout.addWidget(QLabel("Output DEM folder:"), 2, 0)
        self.batch_output_edit = QLineEdit()
        batch_layout.addWidget(self.batch_output_edit, 2, 1)
        batch_output_btn = QPushButton("Browse")
        batch_output_btn.clicked.connect(self.select_batch_output)
        batch_layout.addWidget(batch_output_btn, 2, 2)

        # Guidance text
        batch_info = QLabel(
            "💡 Batch mode: each SAR TIF drives DEM cropping and preprocessing.\n🎯 SAR mask: zero-value SAR pixels are excluded during normalization to avoid boundary artifacts.\n📊 Outputs: every file yields contour statistics, PNG tiles, and Excel/CSV reports.")
        batch_info.setStyleSheet("color: #666; font-size: 11px; margin: 5px;")
        batch_layout.addWidget(batch_info, 3, 0, 1, 3)

        self.batch_file_group.setLayout(batch_layout)
        self.batch_file_group.setVisible(False)  # hidden by default
        layout.addWidget(self.batch_file_group)

        # Action button
        self.process_btn = QPushButton("Start DEM Preprocessing")
        self.process_btn.setStyleSheet("QPushButton { font-size: 14px; font-weight: bold; }")
        self.process_btn.clicked.connect(self.start_processing)
        layout.addWidget(self.process_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Status text
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(150)
        layout.addWidget(self.status_text)

        self.setLayout(layout)

    def toggle_mode(self, button):
        """Toggle processing mode"""
        if self.single_mode_radio.isChecked():
            self.single_file_group.setVisible(True)
            self.batch_file_group.setVisible(False)
            self.process_btn.setText("Start DEM Preprocessing")
        elif self.batch_mode_radio.isChecked():
            self.single_file_group.setVisible(False)
            self.batch_file_group.setVisible(True)
            self.process_btn.setText("Start Batch DEM Preprocessing")

    def select_dem(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select DEM File", "", "TIF Files (*.tif *.tiff)")
        if file_path:
            self.dem_edit.setText(file_path)

    def select_reference(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Reference SAR File", "", "TIF Files (*.tif *.tiff)")
        if file_path:
            self.reference_edit.setText(file_path)

    def select_output(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Select Output File", "", "TIF Files (*.tif)")
        if file_path:
            self.output_edit.setText(file_path)

    def select_batch_dem(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Large DEM File", "", "TIF Files (*.tif *.tiff)")
        if file_path:
            self.batch_dem_edit.setText(file_path)

    def select_sar_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select SAR Folder")
        if folder_path:
            self.sar_folder_edit.setText(folder_path)

    def select_batch_output(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Output DEM Folder")
        if folder_path:
            self.batch_output_edit.setText(folder_path)

    def start_processing(self):
        if self.single_mode_radio.isChecked():
            self.start_single_processing()
        else:
            self.start_batch_processing()

    def start_single_processing(self):
        """Start single-file processing"""
        dem_path = self.dem_edit.text().strip()
        reference_path = self.reference_edit.text().strip()
        output_path = self.output_edit.text().strip()

        if not dem_path or not reference_path or not output_path:
            QMessageBox.warning(self, "Warning", "Select all required files.")
            return

        if not os.path.exists(dem_path):
            QMessageBox.warning(self, "Warning", "DEM file does not exist.")
            return

        if not os.path.exists(reference_path):
            QMessageBox.warning(self, "Warning", "Reference file does not exist.")
            return

        self.process_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_text.clear()

        self.worker_thread = WorkerThread("step3_single",
                                          dem_path=dem_path,
                                          reference_path=reference_path,
                                          output_path=output_path)
        self.worker_thread.progress_updated.connect(self.progress_bar.setValue)
        self.worker_thread.status_updated.connect(self.append_status)
        self.worker_thread.finished_signal.connect(self.on_processing_finished)
        self.worker_thread.start()

    def start_batch_processing(self):
        """Start batch processing"""
        dem_path = self.batch_dem_edit.text().strip()
        sar_folder = self.sar_folder_edit.text().strip()
        output_folder = self.batch_output_edit.text().strip()

        if not dem_path or not sar_folder or not output_folder:
            QMessageBox.warning(self, "Warning", "Select all required files and folders.")
            return

        if not os.path.exists(dem_path):
            QMessageBox.warning(self, "Warning", "Large DEM file does not exist.")
            return

        if not os.path.exists(sar_folder):
            QMessageBox.warning(self, "Warning", "SAR folder does not exist.")
            return

        # Verify SAR folder contains TIFs
        sar_files = []
        for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
            sar_files.extend(glob.glob(os.path.join(sar_folder, ext)))

        if not sar_files:
            QMessageBox.warning(self, "Warning", "No TIF files found in the SAR folder.")
            return

        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)

        self.process_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_text.clear()

        self.worker_thread = WorkerThread("step3_batch",
                                          dem_path=dem_path,
                                          sar_folder=sar_folder,
                                          output_folder=output_folder)
        self.worker_thread.progress_updated.connect(self.progress_bar.setValue)
        self.worker_thread.status_updated.connect(self.append_status)
        self.worker_thread.finished_signal.connect(self.on_processing_finished)
        self.worker_thread.start()

    def append_status(self, message):
        self.status_text.append(message)

    def on_processing_finished(self, success, message):
        self.process_btn.setEnabled(True)

        if success:
            self.append_status(f"✅ {message}")
            QMessageBox.information(self, "Success", message)
        else:
            self.append_status(f"❌ {message}")
            QMessageBox.critical(self, "Error", message)


class Step4Widget(QWidget):
    """Step 4 widget - DoG contour detection"""

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.worker_thread = None

    def init_ui(self):
        # Create scroll area to host content
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        main_widget = QWidget()
        layout = QVBoxLayout()

        # Title
        title = QLabel("Step 4: DoG Contour Detection")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        layout.addWidget(title)

        # Mode selection
        mode_group = QGroupBox("Processing Mode")
        mode_layout = QHBoxLayout()

        # Mutually exclusive radio buttons
        self.mode_button_group = QButtonGroup()

        self.single_mode_radio = QRadioButton("Single File")
        self.single_mode_radio.setChecked(True)
        self.mode_button_group.addButton(self.single_mode_radio, 0)
        mode_layout.addWidget(self.single_mode_radio)

        self.batch_mode_radio = QRadioButton("Batch")
        self.mode_button_group.addButton(self.batch_mode_radio, 1)
        mode_layout.addWidget(self.batch_mode_radio)

        # Connect signals
        self.mode_button_group.buttonClicked.connect(self.toggle_mode)

        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Single-file section
        self.single_file_group = QGroupBox("Single File")
        single_layout = QGridLayout()

        single_layout.addWidget(QLabel("SAR image:"), 0, 0)
        self.sar_edit = QLineEdit()
        single_layout.addWidget(self.sar_edit, 0, 1)
        sar_btn = QPushButton("Browse")
        sar_btn.clicked.connect(self.select_sar)
        single_layout.addWidget(sar_btn, 0, 2)

        single_layout.addWidget(QLabel("DEM file:"), 1, 0)
        self.dem_edit = QLineEdit()
        single_layout.addWidget(self.dem_edit, 1, 1)
        dem_btn = QPushButton("Browse")
        dem_btn.clicked.connect(self.select_dem)
        single_layout.addWidget(dem_btn, 1, 2)

        single_layout.addWidget(QLabel("Output directory:"), 2, 0)
        self.output_edit = QLineEdit()
        single_layout.addWidget(self.output_edit, 2, 1)
        output_btn = QPushButton("Browse")
        output_btn.clicked.connect(self.select_output)
        single_layout.addWidget(output_btn, 2, 2)

        self.single_file_group.setLayout(single_layout)
        layout.addWidget(self.single_file_group)

        # Batch section
        self.batch_file_group = QGroupBox("Batch")
        batch_layout = QGridLayout()

        batch_layout.addWidget(QLabel("SAR folder:"), 0, 0)
        self.sar_folder_edit = QLineEdit()
        batch_layout.addWidget(self.sar_folder_edit, 0, 1)
        sar_folder_btn = QPushButton("Browse")
        sar_folder_btn.clicked.connect(self.select_sar_folder)
        batch_layout.addWidget(sar_folder_btn, 0, 2)

        batch_layout.addWidget(QLabel("DEM mode:"), 1, 0)
        self.dem_mode_combo = QComboBox()
        self.dem_mode_combo.addItems(["Single DEM File", "DEM Folder (One-to-One)"])
        self.dem_mode_combo.currentTextChanged.connect(self.on_dem_mode_changed)
        batch_layout.addWidget(self.dem_mode_combo, 1, 1)

        batch_layout.addWidget(QLabel("DEM file/folder:"), 2, 0)
        self.dem_batch_edit = QLineEdit()
        batch_layout.addWidget(self.dem_batch_edit, 2, 1)
        self.dem_batch_btn = QPushButton("Browse File")
        self.dem_batch_btn.clicked.connect(self.select_dem_batch)
        batch_layout.addWidget(self.dem_batch_btn, 2, 2)

        batch_layout.addWidget(QLabel("Batch output directory:"), 3, 0)
        self.batch_output_edit = QLineEdit()
        batch_layout.addWidget(self.batch_output_edit, 3, 1)
        batch_output_btn = QPushButton("Browse")
        batch_output_btn.clicked.connect(self.select_batch_output)
        batch_layout.addWidget(batch_output_btn, 3, 2)

        # Batch description
        batch_info_step4 = QLabel(
            "📊 Batch DoG detection produces:\n- Every intermediate mask\n- Connected-region PNG tiles\n- DEM-aware Excel/CSV stats\n- Numbered contour visualizations")
        batch_info_step4.setStyleSheet("color: #666; font-size: 11px; margin: 5px;")
        batch_layout.addWidget(batch_info_step4, 4, 0, 1, 3)

        self.batch_file_group.setLayout(batch_layout)
        self.batch_file_group.setVisible(False)  # hidden by default
        layout.addWidget(self.batch_file_group)

        # Parameter configuration
        param_group = QGroupBox("Processing Parameters")
        param_layout = QGridLayout()

        # Tips
        param_info = QLabel("⚙️ These parameters drive the actual processing. Adjust manually or load a preset below.")
        param_info.setStyleSheet("color: #0066cc; font-size: 11px; margin: 5px;")
        param_layout.addWidget(param_info, 0, 0, 1, 4)

        # Edge detection
        param_layout.addWidget(QLabel("Edge threshold:"), 1, 0)
        self.edge_threshold_spin = QSpinBox()
        self.edge_threshold_spin.setRange(1, 255)
        self.edge_threshold_spin.setValue(50)
        param_layout.addWidget(self.edge_threshold_spin, 1, 1)

        # Connected-component filters
        param_layout.addWidget(QLabel("Filter #1 area:"), 2, 0)
        self.threshold1_spin = QSpinBox()
        self.threshold1_spin.setRange(1, 10000)
        self.threshold1_spin.setValue(200)
        param_layout.addWidget(self.threshold1_spin, 2, 1)

        param_layout.addWidget(QLabel("Opening kernel size:"), 2, 2)
        self.kernel1_spin = QSpinBox()
        self.kernel1_spin.setRange(1, 20)
        self.kernel1_spin.setValue(4)
        param_layout.addWidget(self.kernel1_spin, 2, 3)

        param_layout.addWidget(QLabel("Filter #2 area:"), 3, 0)
        self.threshold2_spin = QSpinBox()
        self.threshold2_spin.setRange(1, 10000)
        self.threshold2_spin.setValue(200)
        param_layout.addWidget(self.threshold2_spin, 3, 1)

        param_layout.addWidget(QLabel("Erosion kernel size:"), 3, 2)
        self.kernel2_spin = QSpinBox()
        self.kernel2_spin.setRange(1, 20)
        self.kernel2_spin.setValue(2)
        param_layout.addWidget(self.kernel2_spin, 3, 3)

        param_layout.addWidget(QLabel("Filter #3 area:"), 4, 0)
        self.threshold3_spin = QSpinBox()
        self.threshold3_spin.setRange(1, 10000)
        self.threshold3_spin.setValue(500)
        param_layout.addWidget(self.threshold3_spin, 4, 1)

        # Thinning parameters
        param_layout.addWidget(QLabel("Enable dilation after thinning:"), 5, 0)
        self.enable_dilation_check = QCheckBox()
        param_layout.addWidget(self.enable_dilation_check, 5, 1)

        param_layout.addWidget(QLabel("Dilation kernel size:"), 5, 2)
        self.dilation_kernel_spin = QSpinBox()
        self.dilation_kernel_spin.setRange(1, 10)
        self.dilation_kernel_spin.setValue(2)
        param_layout.addWidget(self.dilation_kernel_spin, 5, 3)

        # Burr removal
        param_layout.addWidget(QLabel("Burr length threshold:"), 6, 0)
        self.burr_threshold_spin = QSpinBox()
        self.burr_threshold_spin.setRange(1, 100)
        self.burr_threshold_spin.setValue(30)
        param_layout.addWidget(self.burr_threshold_spin, 6, 1)

        param_layout.addWidget(QLabel("Small contour threshold:"), 6, 2)
        self.small_contour_spin = QSpinBox()
        self.small_contour_spin.setRange(1, 100)
        self.small_contour_spin.setValue(20)
        param_layout.addWidget(self.small_contour_spin, 6, 3)

        param_layout.addWidget(QLabel("Connected-region padding:"), 7, 0)
        self.padding_spin = QSpinBox()
        self.padding_spin.setRange(0, 20)
        self.padding_spin.setValue(5)
        param_layout.addWidget(self.padding_spin, 7, 1)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # Preset parameter shortcuts
        preset_group = QGroupBox("Presets (optional)")
        preset_layout = QVBoxLayout()

        # Guidance
        preset_info = QLabel("💡 Presets provide quick starts; processing uses the values above.")
        preset_info.setStyleSheet("color: #666; font-size: 11px; margin: 5px;")
        preset_layout.addWidget(preset_info)

        preset_buttons_layout = QHBoxLayout()

        preset1_btn = QPushButton("Preset 1: Standard")
        preset1_btn.clicked.connect(self.load_preset1)
        preset_buttons_layout.addWidget(preset1_btn)

        preset2_btn = QPushButton("Preset 2: Fine")
        preset2_btn.clicked.connect(self.load_preset2)
        preset_buttons_layout.addWidget(preset2_btn)

        preset3_btn = QPushButton("Preset 3: Coarse")
        preset3_btn.clicked.connect(self.load_preset3)
        preset_buttons_layout.addWidget(preset3_btn)

        preset_layout.addLayout(preset_buttons_layout)
        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)

        # Action button
        self.process_btn = QPushButton("Start DoG Contour Detection")
        self.process_btn.setStyleSheet("QPushButton { font-size: 14px; font-weight: bold; }")
        self.process_btn.clicked.connect(self.start_processing)
        layout.addWidget(self.process_btn)

        # Progress bar
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)

        # Status console
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(150)
        layout.addWidget(self.status_text)

        main_widget.setLayout(layout)
        scroll.setWidget(main_widget)

        # Root layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        self.setLayout(main_layout)

        # Initial visibility
        self.single_file_group.setVisible(True)
        self.batch_file_group.setVisible(False)

    def toggle_mode(self, button):
        """Switch between single and batch modes"""
        if self.single_mode_radio.isChecked():
            self.single_file_group.setVisible(True)
            self.batch_file_group.setVisible(False)
            self.process_btn.setText("Start DoG Contour Detection")
        elif self.batch_mode_radio.isChecked():
            self.single_file_group.setVisible(False)
            self.batch_file_group.setVisible(True)
            self.process_btn.setText("Start Batch DoG Contour Detection")

    def on_dem_mode_changed(self):
        """Update DEM selector button when the mode changes"""
        if self.dem_mode_combo.currentText() == "Single DEM File":
            self.dem_batch_btn.setText("Browse File")
        else:
            self.dem_batch_btn.setText("Browse Folder")

    def select_sar(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select SAR Image", "", "TIF Files (*.tif *.tiff)")
        if file_path:
            self.sar_edit.setText(file_path)

    def select_dem(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select DEM File", "", "TIF Files (*.tif *.tiff)")
        if file_path:
            self.dem_edit.setText(file_path)

    def select_output(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if folder_path:
            self.output_edit.setText(folder_path)

    def select_sar_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select SAR Image Folder")
        if folder_path:
            self.sar_folder_edit.setText(folder_path)

    def select_dem_batch(self):
        if self.dem_mode_combo.currentText() == "Single DEM File":
            file_path, _ = QFileDialog.getOpenFileName(self, "Select DEM File", "", "TIF Files (*.tif *.tiff)")
            if file_path:
                self.dem_batch_edit.setText(file_path)
        else:
            folder_path = QFileDialog.getExistingDirectory(self, "Select DEM Folder")
            if folder_path:
                self.dem_batch_edit.setText(folder_path)

    def select_batch_output(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Batch Output Directory")
        if folder_path:
            self.batch_output_edit.setText(folder_path)

    def load_preset1(self):
        """Standard preset"""
        self.edge_threshold_spin.setValue(50)
        self.threshold1_spin.setValue(200)
        self.kernel1_spin.setValue(4)
        self.threshold2_spin.setValue(200)
        self.kernel2_spin.setValue(2)
        self.threshold3_spin.setValue(500)
        self.enable_dilation_check.setChecked(False)
        self.dilation_kernel_spin.setValue(2)
        self.burr_threshold_spin.setValue(30)
        self.small_contour_spin.setValue(20)
        self.padding_spin.setValue(5)

    def load_preset2(self):
        """Fine preset"""
        self.edge_threshold_spin.setValue(30)
        self.threshold1_spin.setValue(100)
        self.kernel1_spin.setValue(3)
        self.threshold2_spin.setValue(100)
        self.kernel2_spin.setValue(1)
        self.threshold3_spin.setValue(300)
        self.enable_dilation_check.setChecked(True)
        self.dilation_kernel_spin.setValue(1)
        self.burr_threshold_spin.setValue(20)
        self.small_contour_spin.setValue(10)
        self.padding_spin.setValue(3)

    def load_preset3(self):
        """Coarse preset"""
        self.edge_threshold_spin.setValue(80)
        self.threshold1_spin.setValue(500)
        self.kernel1_spin.setValue(6)
        self.threshold2_spin.setValue(500)
        self.kernel2_spin.setValue(3)
        self.threshold3_spin.setValue(1000)
        self.enable_dilation_check.setChecked(False)
        self.dilation_kernel_spin.setValue(3)
        self.burr_threshold_spin.setValue(50)
        self.small_contour_spin.setValue(40)
        self.padding_spin.setValue(8)

    def get_parameters(self):
        """Return the active parameter set"""
        return {
            'edge_threshold': self.edge_threshold_spin.value(),
            'threshold1': self.threshold1_spin.value(),
            'kernel1_size': self.kernel1_spin.value(),
            'threshold2': self.threshold2_spin.value(),
            'kernel2_size': self.kernel2_spin.value(),
            'threshold3': self.threshold3_spin.value(),
            'enable_dilation': self.enable_dilation_check.isChecked(),
            'dilation_kernel_size': self.dilation_kernel_spin.value(),
            'burr_threshold': self.burr_threshold_spin.value(),
            'small_contour_threshold': self.small_contour_spin.value(),
            'padding': self.padding_spin.value()
        }

    def start_processing(self):
        if self.single_mode_radio.isChecked():
            self.start_single_processing()
        else:
            self.start_batch_processing()

    def start_single_processing(self):
        """Run DoG detection for a single file"""
        sar_path = self.sar_edit.text().strip()
        dem_path = self.dem_edit.text().strip()
        output_dir = self.output_edit.text().strip()

        if not sar_path or not dem_path or not output_dir:
            QMessageBox.warning(self, "Warning", "Select all required files and directories.")
            return

        if not os.path.exists(sar_path):
            QMessageBox.warning(self, "Warning", "SAR image file does not exist.")
            return

        if not os.path.exists(dem_path):
            QMessageBox.warning(self, "Warning", "DEM file does not exist.")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        parameters = self.get_parameters()

        self.process_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_text.clear()

        self.worker_thread = WorkerThread("step4_single",
                                          sar_path=sar_path,
                                          dem_path=dem_path,
                                          output_dir=output_dir,
                                          parameters=parameters)
        self.worker_thread.progress_updated.connect(self.progress_bar.setValue)
        self.worker_thread.status_updated.connect(self.append_status)
        self.worker_thread.finished_signal.connect(self.on_processing_finished)
        self.worker_thread.start()

    def start_batch_processing(self):
        """Run DoG detection in batch mode"""
        sar_folder = self.sar_folder_edit.text().strip()
        dem_path_or_folder = self.dem_batch_edit.text().strip()
        output_dir = self.batch_output_edit.text().strip()
        dem_mode = self.dem_mode_combo.currentText()

        if not sar_folder or not dem_path_or_folder or not output_dir:
            QMessageBox.warning(self, "Warning", "Select all required folders and directories.")
            return

        if not os.path.exists(sar_folder):
            QMessageBox.warning(self, "Warning", "SAR image folder does not exist.")
            return

        if not os.path.exists(dem_path_or_folder):
            QMessageBox.warning(self, "Warning", "DEM file/folder does not exist.")
            return

        # Ensure the SAR folder has TIF files
        sar_files = []
        for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
            sar_files.extend(glob.glob(os.path.join(sar_folder, ext)))

        if not sar_files:
            QMessageBox.warning(self, "Warning", "No TIF files were found in the SAR folder.")
            return

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        parameters = self.get_parameters()

        self.process_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_text.clear()

        self.worker_thread = WorkerThread("step4_batch",
                                          sar_folder=sar_folder,
                                          dem_path_or_folder=dem_path_or_folder,
                                          output_dir=output_dir,
                                          dem_mode=dem_mode,
                                          parameters=parameters)
        self.worker_thread.progress_updated.connect(self.progress_bar.setValue)
        self.worker_thread.status_updated.connect(self.append_status)
        self.worker_thread.finished_signal.connect(self.on_processing_finished)
        self.worker_thread.start()

    def append_status(self, message):
        self.status_text.append(message)

    def on_processing_finished(self, success, message):
        self.process_btn.setEnabled(True)

        if success:
            self.append_status(f"✅ {message}")
            QMessageBox.information(self, "Success", message)
        else:
            self.append_status(f"❌ {message}")
            QMessageBox.critical(self, "Error", message)


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("SAR & DEM Processing Suite")
        self.setGeometry(100, 100, 900, 700)

        # Tab widget aggregates the workflow steps
        tab_widget = QTabWidget()

        # Add the four workflow steps
        tab_widget.addTab(Step1Widget(), "Step 1: SAR Preprocessing")
        tab_widget.addTab(Step2Widget(), "Step 2: Geolocation Adjustment")
        tab_widget.addTab(Step3Widget(), "Step 3: DEM Preprocessing")
        tab_widget.addTab(Step4Widget(), "Step 4: DoG Contour Detection")

        self.setCentralWidget(tab_widget)


def main():
    app = QApplication(sys.argv)

    # Apply application style
    app.setStyle('Fusion')

    # Create and display the main window
    window = MainWindow()
    window.show()

    # Run the Qt event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
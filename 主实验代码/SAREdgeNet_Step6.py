import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from saredge.models import SAREdgeNetContourClassifier
'\nLake Detection System - Simplified Edition\nSecurity guards removed, core functionality retained\n\nAuthor: AI Assistant\nVersion: 1.2 (Simplified) – fixes the PIL max image size limit\n'
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
import gc
import cv2
import warnings
import traceback
Image.MAX_IMAGE_PIXELS = None
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel, QLineEdit, QPushButton, QTextEdit, QFileDialog, QProgressBar, QMessageBox, QGroupBox, QGridLayout, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QIcon
import rasterio
from rasterio.transform import from_bounds
from osgeo import gdal, osr
import seaborn as sns
warnings.filterwarnings('ignore')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
torch.backends.cudnn.benchmark = False

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

class GrayscaleDataset(Dataset):

    def __init__(self, data_path, image_dir, transform=None, feature_scaler=None):
        if data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path, encoding='utf-8-sig')
        else:
            self.data = pd.read_excel(data_path)
        self.image_dir = image_dir
        self.transform = transform
        self.feature_scaler = feature_scaler
        self.image_names = self.data.iloc[:, 0].values
        self.features = self.data.iloc[:, 1:-1].values.astype(np.float32)
        self.labels = self.data.iloc[:, -1].values.astype(np.int64)
        if self.feature_scaler is not None:
            self.features = self.feature_scaler.transform(self.features)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_name = f'{self.image_names[idx]}.png'
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert('L')
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
        if self.transform:
            image = self.transform(image)
        features = torch.FloatTensor(self.features[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return (image, features, label)

def get_grayscale_transforms():
    train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(p=0.3), transforms.RandomRotation(10), transforms.ToTensor(), transforms.Normalize(mean=[0.485], std=[0.229])])
    val_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485], std=[0.229])])
    return (train_transform, val_transform)

def get_prediction_transform():
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485], std=[0.229])])

def calculate_class_weights(labels):
    class_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(class_counts)
    class_weights = {}
    for class_id in sorted(class_counts.keys()):
        class_weights[class_id] = total_samples / (num_classes * class_counts[class_id])
    weights_tensor = torch.FloatTensor([class_weights[i] for i in sorted(class_weights.keys())])
    return weights_tensor

def calculate_confusion_matrix(all_labels, all_predictions, class_names=['Class0', 'Class1']):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel()
    precision_0 = tn / (tn + fn) if tn + fn > 0 else 0
    recall_0 = tn / (tn + fp) if tn + fp > 0 else 0
    f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if precision_0 + recall_0 > 0 else 0
    precision_1 = tp / (tp + fp) if tp + fp > 0 else 0
    recall_1 = tp / (tp + fn) if tp + fn > 0 else 0
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if precision_1 + recall_1 > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else 0
    matrix_str = f'\n╔═══════════════════════════════════════════════════════════════╗\n║              Grayscale Image Confusion Matrix Analysis        ║\n╠═══════════════════════════════════════════════════════════════╣\n║                        Prediction Results                     ║\n║          ┌─────────────┬─────────────┬─────────────┐          ║\n║  True    │             │   {class_names[0]:^9}   │   {class_names[1]:^9}   │          ║\n║          ├─────────────┼─────────────┼─────────────┤          ║\n║          │  {class_names[0]:^9}  │    {tn:^7}    │    {fp:^7}    │          ║\n║          ├─────────────┼─────────────┼─────────────┤          ║\n║          │  {class_names[1]:^9}  │    {fn:^7}    │    {tp:^7}    │          ║\n║          └─────────────┴─────────────┴─────────────┘          ║\n╠═══════════════════════════════════════════════════════════════╣\n║  Metric Details:                                              ║\n║                                                               ║\n║  Overall Accuracy: {accuracy:6.2%}                                      ║\n║                                                               ║\n║  {class_names[0]} (Negative):                                        ║\n║    Precision: {precision_0:6.2%}   Recall: {recall_0:6.2%}   F1-Score: {f1_0:6.2%}    ║\n║                                                               ║\n║  {class_names[1]} (Positive):                                        ║\n║    Precision: {precision_1:6.2%}   Recall: {recall_1:6.2%}   F1-Score: {f1_1:6.2%}    ║\n║                                                               ║\n║  Confusion Matrix Legend:                                     ║\n║    TN(True Negative): {tn:4d}  |  FP(False Positive): {fp:4d}                       ║\n║    FN(False Negative): {fn:4d}  |  TP(True Positive): {tp:4d}                       ║\n╚═══════════════════════════════════════════════════════════════╝\n'
    return matrix_str

def save_confusion_matrix_plot(all_labels, all_predictions, epoch, output_dir, class_names=['Class0', 'Class1']):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Grayscale Image Confusion Matrix - Epoch {epoch + 1}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(output_dir, f'grayscale_confusion_matrix_epoch_{epoch + 1}.png'), dpi=150, bbox_inches='tight')
    plt.close()

def read_geotiff_info(image_path):
    try:
        with rasterio.open(image_path) as dataset:
            geotransform = dataset.transform
            projection = dataset.crs
            return (geotransform, projection)
    except:
        dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
        geotransform = dataset.GetGeoTransform()
        projection = dataset.GetProjection()
        dataset = None
        return (geotransform, projection)

def save_array_as_geotiff(array, output_path, geotransform=None, projection=None, datatype=None):
    if datatype is None:
        datatype = gdal.GDT_Byte if array.dtype == np.uint8 else gdal.GDT_Float32
    if len(array.shape) == 3:
        height, width, bands = array.shape
    else:
        height, width = array.shape
        bands = 1
        array = array[:, :, np.newaxis]
    try:
        if geotransform is not None and hasattr(geotransform, 'to_gdal'):
            transform = geotransform
            crs = projection
        else:
            if geotransform is not None:
                transform = rasterio.transform.Affine.from_gdal(*geotransform)
            else:
                transform = None
            crs = projection
        with rasterio.open(output_path, 'w', driver='GTiff', height=height, width=width, count=bands, dtype=array.dtype, crs=crs, transform=transform) as dst:
            if bands == 1:
                dst.write(array[:, :, 0], 1)
            else:
                for i in range(bands):
                    dst.write(array[:, :, i], i + 1)
        return True
    except:
        try:
            driver = gdal.GetDriverByName('GTiff')
            dataset = driver.Create(output_path, width, height, bands, datatype)
            if geotransform is not None:
                if hasattr(geotransform, 'to_gdal'):
                    dataset.SetGeoTransform(geotransform.to_gdal())
                else:
                    dataset.SetGeoTransform(geotransform)
            if projection is not None:
                if hasattr(projection, 'to_string'):
                    dataset.SetProjection(projection.to_string())
                else:
                    dataset.SetProjection(str(projection))
            if bands == 1:
                dataset.GetRasterBand(1).WriteArray(array[:, :, 0])
            else:
                for i in range(bands):
                    dataset.GetRasterBand(i + 1).WriteArray(array[:, :, i])
            dataset = None
            return True
        except:
            if len(array.shape) == 2:
                Image.fromarray(array).save(output_path)
            else:
                Image.fromarray(array).save(output_path)
            return False

def load_model_and_scaler(model_path, statistics_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = SAREdgeNetContourClassifier(pretrained=False)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device).eval()
    scaler_path = Path(model_path).parent / 'feature_scaler.npz'
    saved = np.load(scaler_path, allow_pickle=False)
    data = pd.read_csv(statistics_path) if str(statistics_path).endswith('.csv') else pd.read_excel(statistics_path)
    expected = saved['feature_columns'].tolist()
    if list(data.columns[1:-1]) != expected:
        raise ValueError('Prediction features differ from training features/order')
    scaler = StandardScaler()
    scaler.mean_, scaler.scale_ = (saved['mean'], saved['scale'])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = 17
    return (model, scaler, device)

def predict_single_contour(model, image_path, features, transform, scaler, device):
    if image_path.lower().endswith(('.tif', '.tiff')):
        image = Image.open(image_path).convert('L')
    else:
        image = Image.open(image_path).convert('L')
    image_tensor = transform(image).unsqueeze(0).to(device)
    features_scaled = scaler.transform([features])[0]
    features_tensor = torch.FloatTensor(features_scaled).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor, features_tensor)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    return (predicted_class, confidence)

def create_colored_contour(original_image_path, predicted_class, confidence, output_dir, geotransform=None, projection=None):
    try:
        if original_image_path.lower().endswith(('.tif', '.tiff')):
            with rasterio.open(original_image_path) as src:
                if src.count == 1:
                    contour_img = src.read(1)
                else:
                    contour_img = src.read(1)
        else:
            contour_img = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        if contour_img is None:
            print(f'⚠️ Unable to read image: {original_image_path}')
            return (None, None)
        colored_img = cv2.cvtColor(contour_img, cv2.COLOR_GRAY2BGR)
        if predicted_class == 1:
            color = (0, 0, 255)
            color_name = 'Red'
        else:
            color = (255, 0, 0)
            color_name = 'Blue'
        white_pixels = contour_img == 255
        colored_img[white_pixels] = color
        base_name = os.path.basename(original_image_path)
        name_without_ext = os.path.splitext(base_name)[0]
        colored_path = os.path.join(output_dir, f'colored_{name_without_ext}.tif')
        save_array_as_geotiff(colored_img, colored_path, geotransform, projection)
        print(f'  Contour {base_name}: Class {predicted_class} ({color_name}), Confidence {confidence:.3f}')
        return (colored_img, color)
    except Exception as e:
        print(f'⚠️ Error processing contour image: {e}')
        return (None, None)

def calculate_actual_square_positions(statistics_path, positions_path):
    print(f'📄 Calculating actual square positions (fixing alignment issues)...')
    PADDING = 5
    if statistics_path.endswith('.csv'):
        stats_df = pd.read_csv(statistics_path, encoding='utf-8-sig')
    else:
        stats_df = pd.read_excel(statistics_path)
    if positions_path.endswith('.csv'):
        positions_df = pd.read_csv(positions_path, encoding='utf-8-sig')
    else:
        positions_df = pd.read_excel(positions_path)
    print(f'   Statistics records: {len(stats_df)}')
    print(f'   Position records: {len(positions_df)}')
    merged_data = {}
    for idx, row in stats_df.iterrows():
        region_id = int(row.iloc[0])
        w = int(row.iloc[2])
        h = int(row.iloc[3])
        merged_data[region_id] = {'w': w, 'h': h}
    corrected_positions = []
    for idx, row in positions_df.iterrows():
        region_id = int(row.iloc[0])
        original_x = int(row.iloc[1])
        original_y = int(row.iloc[2])
        if region_id in merged_data:
            w = merged_data[region_id]['w']
            h = merged_data[region_id]['h']
            max_dimension = max(w, h)
            center_x = original_x + w // 2
            center_y = original_y + h // 2
            half_size = (max_dimension + 2 * PADDING) // 2
            square_x_min = max(0, center_x - half_size)
            square_y_min = max(0, center_y - half_size)
            corrected_positions.append({'id': region_id, 'original_x': original_x, 'original_y': original_y, 'actual_x': square_x_min, 'actual_y': square_y_min, 'w': w, 'h': h, 'square_size': 2 * half_size})
            if len(corrected_positions) <= 5:
                print(f'   Region {region_id}: Original position ({original_x},{original_y}) Size ({w}x{h}) -> Actual position ({square_x_min},{square_y_min})')
    print(f'✅ Successfully calculated actual positions for {len(corrected_positions)} regions')
    return corrected_positions

def overlay_contours_on_original(original_image_path, positions_data, colored_contours_dir, output_path, geotransform=None, projection=None, filter_class=None):
    filter_desc = 'All contours' if filter_class is None else f'Only class {filter_class} ({('Red' if filter_class == 1 else 'Blue')}) contours'
    print(f'📄 Starting to overlay contours onto original image ({filter_desc})...')
    try:
        with rasterio.open(original_image_path) as src:
            print(f'   Reading large SAR image: {original_image_path}')
            print(f'   Image size: {src.width} x {src.height}, Band count: {src.count}')
            if src.count >= 3:
                original_img = src.read([1, 2, 3]).transpose(1, 2, 0)
                original_img = np.uint8(original_img)
            else:
                band_data = src.read(1)
                if band_data.dtype != np.uint8:
                    band_min, band_max = np.percentile(band_data, [2, 98])
                    band_data = np.clip((band_data - band_min) / (band_max - band_min) * 255, 0, 255)
                    band_data = band_data.astype(np.uint8)
                original_img = np.stack([band_data, band_data, band_data], axis=2)
    except Exception as e:
        print(f'❌ Failed to read with rasterio: {e}')
        try:
            if original_image_path.lower().endswith('.tif'):
                original_img_pil = Image.open(original_image_path)
                if original_img_pil.mode != 'RGB':
                    original_img_pil = original_img_pil.convert('RGB')
                original_img = cv2.cvtColor(np.array(original_img_pil), cv2.COLOR_RGB2BGR)
            else:
                original_img = cv2.imread(original_image_path)
        except Exception as e2:
            print(f'❌ Fallback also failed: {e2}')
            raise ValueError(f'Unable to read original image: {original_image_path}')
    if original_img is None:
        raise ValueError(f'Unable to read original image: {original_image_path}')
    result_img = original_img.copy()
    print(f'   Original image size: {original_img.shape[:2]}')
    print(f'   Contour count: {len(positions_data)}')
    overlay_count = 0
    class_counts = {'Class0(Blue)': 0, 'Class1(Red)': 0}
    for position_info in positions_data:
        region_id = position_info['id']
        actual_x = position_info['actual_x']
        actual_y = position_info['actual_y']
        colored_contour_path = os.path.join(colored_contours_dir, f'colored_{region_id}.tif')
        if os.path.exists(colored_contour_path):
            if colored_contour_path.lower().endswith('.tif'):
                try:
                    with rasterio.open(colored_contour_path) as contour_src:
                        if contour_src.count >= 3:
                            colored_contour = contour_src.read([1, 2, 3]).transpose(1, 2, 0)
                        else:
                            band_data = contour_src.read(1)
                            colored_contour = np.stack([band_data, band_data, band_data], axis=2)
                        colored_contour = np.uint8(colored_contour)
                        colored_contour = cv2.cvtColor(colored_contour, cv2.COLOR_RGB2BGR)
                except:
                    colored_contour_pil = Image.open(colored_contour_path)
                    if colored_contour_pil.mode != 'RGB':
                        colored_contour_pil = colored_contour_pil.convert('RGB')
                    colored_contour = cv2.cvtColor(np.array(colored_contour_pil), cv2.COLOR_RGB2BGR)
            else:
                colored_contour = cv2.imread(colored_contour_path)
            if colored_contour is not None:
                contour_h, contour_w = colored_contour.shape[:2]
                temp_mask = np.any(colored_contour != [0, 0, 0], axis=2)
                if np.any(temp_mask):
                    avg_color = np.mean(colored_contour[temp_mask], axis=0)
                    contour_class = 1 if avg_color[2] > avg_color[0] else 0
                    should_overlay = filter_class is None or filter_class == contour_class
                    if should_overlay:
                        end_x = min(actual_x + contour_w, original_img.shape[1])
                        end_y = min(actual_y + contour_h, original_img.shape[0])
                        actual_w = end_x - actual_x
                        actual_h = end_y - actual_y
                        if actual_w > 0 and actual_h > 0:
                            contour_region = colored_contour[:actual_h, :actual_w]
                            mask = np.any(contour_region != [0, 0, 0], axis=2)
                            result_img[actual_y:actual_y + actual_h, actual_x:actual_x + actual_w][mask] = contour_region[mask]
                            overlay_count += 1
                            if contour_class == 1:
                                class_counts['Class1(Red)'] += 1
                            else:
                                class_counts['Class0(Blue)'] += 1
                            if overlay_count % 50 == 0:
                                print(f'   Processed {overlay_count} contours...')
    print(f'   Saving large result image...')
    save_array_as_geotiff(result_img, output_path, geotransform, projection)
    print(f'✅ Contour overlay complete ({filter_desc}):')
    print(f'   Successfully overlaid contours: {overlay_count}')
    print(f'   Class statistics: {class_counts}')
    print(f'   Result image saved: {output_path}')
    return (result_img, class_counts)

class BatchWorker(QThread):
    progress_updated = pyqtSignal(int)
    log_updated = pyqtSignal(str)
    finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    current_file_updated = pyqtSignal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.should_stop = False
        self.original_pil_limit = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = None

    def __del__(self):
        if hasattr(self, 'original_pil_limit') and self.original_pil_limit is not None:
            Image.MAX_IMAGE_PIXELS = self.original_pil_limit

    def stop(self):
        self.should_stop = True

    def run(self):
        self.batch_predict()

    def batch_predict(self):
        try:
            self.log_updated.emit('🚀 Starting batch prediction processing...')
            sar_dir = self.config['sar_dir']
            contours_dir = self.config['contours_dir']
            output_dir = self.config['output_dir']
            model_path = self.config['model_path']
            os.makedirs(output_dir, exist_ok=True)
            sar_files = []
            for file in os.listdir(sar_dir):
                if file.endswith('.tif') and 'normalized_processed' in file:
                    sar_files.append(file)
            self.log_updated.emit(f'📂 Found {len(sar_files)} SAR files')
            if len(sar_files) == 0:
                self.log_updated.emit('❌ No SAR files found')
                self.finished.emit()
                return
            successful_count = 0
            failed_count = 0
            for i, sar_file in enumerate(sar_files):
                if self.should_stop:
                    self.log_updated.emit('❌ User stopped batch processing')
                    break
                self.current_file_updated.emit(f'Processing: {sar_file}')
                self.log_updated.emit(f'\n{'=' * 60}')
                self.log_updated.emit(f'📄 Processing file {i + 1}/{len(sar_files)}: {sar_file}')
                try:
                    sar_path = os.path.join(sar_dir, sar_file)
                    base_name = sar_file.replace('.tif', '')
                    contour_folder = os.path.join(contours_dir, base_name)
                    if not os.path.exists(contour_folder):
                        self.log_updated.emit(f'⚠️  Skipping {sar_file}: Corresponding contour folder does not exist')
                        failed_count += 1
                        continue
                    file_output_dir = os.path.join(output_dir, base_name)
                    os.makedirs(file_output_dir, exist_ok=True)
                    positions_path = os.path.join(contour_folder, 'contour_positions.csv')
                    statistics_path = os.path.join(contour_folder, 'contour_statistics_enhanced_with_dem.csv')
                    if not os.path.exists(positions_path):
                        self.log_updated.emit(f'⚠️  Skipping {sar_file}: Position file does not exist')
                        failed_count += 1
                        continue
                    if not os.path.exists(statistics_path):
                        self.log_updated.emit(f'⚠️  Skipping {sar_file}: Statistics file does not exist')
                        failed_count += 1
                        continue
                    success = self.process_single_file(sar_path, contour_folder, positions_path, statistics_path, file_output_dir, model_path)
                    if success:
                        successful_count += 1
                        self.log_updated.emit(f'✅ {sar_file} processing completed')
                    else:
                        failed_count += 1
                        self.log_updated.emit(f'❌ {sar_file} processing failed')
                except Exception as e:
                    failed_count += 1
                    self.log_updated.emit(f'❌ Error processing {sar_file}: {str(e)}')
                progress = int((i + 1) / len(sar_files) * 100)
                self.progress_updated.emit(progress)
            self.log_updated.emit(f'\n{'=' * 60}')
            self.log_updated.emit('🎉 Batch processing completed!')
            self.log_updated.emit(f'📊 Processing statistics:')
            self.log_updated.emit(f'   ✅ Success: {successful_count} files')
            self.log_updated.emit(f'   ❌ Failed: {failed_count} files')
            self.log_updated.emit(f'   📁 Total: {len(sar_files)} files')
            self.log_updated.emit(f'📂 Results saved to: {output_dir}')
            self.finished.emit()
        except Exception as e:
            self.error_occurred.emit(str(e))

    def process_single_file(self, sar_path, contour_folder, positions_path, statistics_path, file_output_dir, model_path):
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            colored_contours_dir = os.path.join(file_output_dir, 'colored_contours')
            os.makedirs(colored_contours_dir, exist_ok=True)
            final_result_path = os.path.join(file_output_dir, 'final_prediction_result.tif')
            red_only_result_path = os.path.join(file_output_dir, 'red_contours_only.tif')
            blue_only_result_path = os.path.join(file_output_dir, 'blue_contours_only.tif')
            geotransform, projection = read_geotiff_info(sar_path)
            model, scaler, device = load_model_and_scaler(model_path, statistics_path)
            transform = get_prediction_transform()
            if statistics_path.endswith('.csv'):
                statistics_df = pd.read_csv(statistics_path, encoding='utf-8-sig')
            else:
                statistics_df = pd.read_excel(statistics_path)
            positions_data = calculate_actual_square_positions(statistics_path, positions_path)
            predictions_log = []
            successful_predictions = 0
            total_contours = len(statistics_df)
            for idx, row in statistics_df.iterrows():
                if self.should_stop:
                    return False
                tif_id = int(row.iloc[0])
                features = row.iloc[1:-1].values.astype(np.float32)
                tif_path = os.path.join(contour_folder, f'{tif_id}.tif')
                if not os.path.exists(tif_path):
                    tif_path = os.path.join(contour_folder, f'{tif_id}.png')
                if os.path.exists(tif_path):
                    predicted_class, confidence = predict_single_contour(model, tif_path, features, transform, scaler, device)
                    colored_result = create_colored_contour(tif_path, predicted_class, confidence, colored_contours_dir, geotransform, projection)
                    if colored_result[0] is not None:
                        successful_predictions += 1
                        predictions_log.append({'TIF_ID': tif_id, 'Predicted_Class': predicted_class, 'Confidence': confidence, 'Color': 'Red' if predicted_class == 1 else 'Blue'})
                if idx % 10 == 0:
                    clear_gpu_memory()
            if predictions_log:
                predictions_df = pd.DataFrame(predictions_log)
                predictions_csv_path = os.path.join(file_output_dir, 'prediction_results.csv')
                predictions_df.to_csv(predictions_csv_path, index=False, encoding='utf-8-sig')
            final_result, class_counts = overlay_contours_on_original(sar_path, positions_data, colored_contours_dir, final_result_path, geotransform, projection, filter_class=None)
            red_result, red_counts = overlay_contours_on_original(sar_path, positions_data, colored_contours_dir, red_only_result_path, geotransform, projection, filter_class=1)
            blue_result, blue_counts = overlay_contours_on_original(sar_path, positions_data, colored_contours_dir, blue_only_result_path, geotransform, projection, filter_class=0)
            clear_gpu_memory()
            return True
        except Exception as e:
            self.log_updated.emit(f'   Error processing file: {str(e)}')
            return False

class LakeDetectionMainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Lake Detection System - Fixed PIL Image Limit Version')
        self.setGeometry(100, 100, 1200, 800)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        title_label = QLabel('Lake Detection System (Fixed PIL Image Limit Version)')
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setStyleSheet('color: #2E86AB; margin: 10px;')
        layout.addWidget(title_label)
        fix_label = QLabel('🔧 Fixed PIL image size limit issue, now can process large SAR images!')
        fix_label.setStyleSheet('background-color: #d4edda; color: #155724; padding: 8px; border-radius: 4px; margin: 5px;')
        layout.addWidget(fix_label)
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        self.create_batch_tab()
        self.statusBar().showMessage('Ready - Fixed PIL image limit, supports large SAR image processing')

    def create_batch_tab(self):
        batch_widget = QWidget()
        self.tab_widget.addTab(batch_widget, 'Batch Processing - SAR Image Classification')
        layout = QVBoxLayout(batch_widget)
        feature_group = QGroupBox('🚀 Batch Processing Features')
        feature_layout = QVBoxLayout(feature_group)
        feature_label = QLabel('\n<b style="color: #d32f2f;">Batch SAR Image Lake Contour Classification:</b><br>\n📂 <b>Auto Scan</b>: Automatically scan SAR image folders and contour folders<br>\n🔍 <b>Smart Matching</b>: Automatically match SAR files with corresponding contour folders<br>\n🔮 <b>Batch Prediction</b>: Classify lake contours for all matched files<br>\n🎨 <b>Result Output</b>: Generate colored contour overlay images and statistics reports<br>\n📊 <b>Progress Monitoring</b>: Real-time display of processing progress and detailed logs<br>\n🔧 <b>Fixed PIL Limit</b>: Now can process SAR images of any size\n        ')
        feature_label.setWordWrap(True)
        feature_label.setStyleSheet('background-color: #e8f5e8; padding: 10px; border-radius: 5px;')
        feature_layout.addWidget(feature_label)
        layout.addWidget(feature_group)
        input_group = QGroupBox('Input Path Settings')
        input_layout = QGridLayout(input_group)
        input_layout.addWidget(QLabel('SAR Image Folder:'), 0, 0)
        self.batch_sar_dir = QLineEdit()
        self.batch_sar_dir.setText('data/sar_tiles')
        self.batch_sar_dir.setPlaceholderText('Select folder containing SAR images')
        input_layout.addWidget(self.batch_sar_dir, 0, 1)
        browse_sar_btn = QPushButton('Browse')
        browse_sar_btn.clicked.connect(self.browse_batch_sar_dir)
        input_layout.addWidget(browse_sar_btn, 0, 2)
        input_layout.addWidget(QLabel('Contour Folder:'), 1, 0)
        self.batch_contours_dir = QLineEdit()
        self.batch_contours_dir.setText('outputs/candidates')
        self.batch_contours_dir.setPlaceholderText('Select folder containing contours')
        input_layout.addWidget(self.batch_contours_dir, 1, 1)
        browse_contours_btn = QPushButton('Browse')
        browse_contours_btn.clicked.connect(self.browse_batch_contours_dir)
        input_layout.addWidget(browse_contours_btn, 1, 2)
        input_layout.addWidget(QLabel('Trained Model:'), 2, 0)
        self.batch_model_path = QLineEdit()
        self.batch_model_path.setPlaceholderText('Select trained .pth model file')
        input_layout.addWidget(self.batch_model_path, 2, 1)
        browse_batch_model_btn = QPushButton('Browse')
        browse_batch_model_btn.clicked.connect(self.browse_batch_model)
        input_layout.addWidget(browse_batch_model_btn, 2, 2)
        layout.addWidget(input_group)
        output_group = QGroupBox('Output Settings')
        output_layout = QGridLayout(output_group)
        output_layout.addWidget(QLabel('Output Directory:'), 0, 0)
        self.batch_output_dir = QLineEdit()
        self.batch_output_dir.setText('outputs/predictions')
        self.batch_output_dir.setPlaceholderText('Select batch processing result output directory')
        output_layout.addWidget(self.batch_output_dir, 0, 1)
        browse_batch_output_btn = QPushButton('Browse')
        browse_batch_output_btn.clicked.connect(self.browse_batch_output_dir)
        output_layout.addWidget(browse_batch_output_btn, 0, 2)
        layout.addWidget(output_group)
        preview_group = QGroupBox('Preview and Check')
        preview_layout = QVBoxLayout(preview_group)
        check_btn = QPushButton('🔍 Check File Matching')
        check_btn.setStyleSheet('QPushButton { background-color: #2196F3; color: white; font-weight: bold; padding: 8px; }')
        check_btn.clicked.connect(self.check_file_matching)
        preview_layout.addWidget(check_btn)
        self.batch_preview = QTextEdit()
        self.batch_preview.setMaximumHeight(100)
        self.batch_preview.setPlaceholderText("Click 'Check File Matching' to view files to be processed...")
        preview_layout.addWidget(self.batch_preview)
        layout.addWidget(preview_group)
        control_layout = QHBoxLayout()
        self.batch_start_btn = QPushButton('🚀 Start Batch Processing')
        self.batch_start_btn.setStyleSheet('QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 10px; }')
        self.batch_start_btn.clicked.connect(self.start_batch_processing)
        control_layout.addWidget(self.batch_start_btn)
        self.batch_stop_btn = QPushButton('Stop Processing')
        self.batch_stop_btn.setStyleSheet('QPushButton { background-color: #f44336; color: white; font-weight: bold; padding: 10px; }')
        self.batch_stop_btn.setEnabled(False)
        self.batch_stop_btn.clicked.connect(self.stop_batch_processing)
        control_layout.addWidget(self.batch_stop_btn)
        layout.addLayout(control_layout)
        progress_group = QGroupBox('Processing Progress')
        progress_layout = QVBoxLayout(progress_group)
        self.batch_current_file = QLabel('Waiting to start...')
        self.batch_current_file.setStyleSheet('color: #666; font-weight: bold;')
        progress_layout.addWidget(self.batch_current_file)
        self.batch_progress = QProgressBar()
        progress_layout.addWidget(self.batch_progress)
        layout.addWidget(progress_group)
        log_group = QGroupBox('Batch Processing Log')
        log_layout = QVBoxLayout(log_group)
        self.batch_log = QTextEdit()
        self.batch_log.setMaximumHeight(200)
        log_layout.addWidget(self.batch_log)
        layout.addWidget(log_group)
        self.batch_worker = None

    def browse_batch_sar_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select SAR Image Folder')
        if dir_path:
            self.batch_sar_dir.setText(dir_path)

    def browse_batch_contours_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select Contour Folder')
        if dir_path:
            self.batch_contours_dir.setText(dir_path)

    def browse_batch_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select Trained Model', '', 'PyTorch models (*.pth *.pt)')
        if file_path:
            self.batch_model_path.setText(file_path)

    def browse_batch_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, 'Select Batch Output Directory')
        if dir_path:
            self.batch_output_dir.setText(dir_path)

    def check_file_matching(self):
        sar_dir = self.batch_sar_dir.text().strip()
        contours_dir = self.batch_contours_dir.text().strip()
        if not sar_dir or not contours_dir:
            self.batch_preview.setText('Please select SAR image folder and contour folder first')
            return
        if not os.path.exists(sar_dir):
            self.batch_preview.setText('SAR image folder does not exist')
            return
        if not os.path.exists(contours_dir):
            self.batch_preview.setText('Contour folder does not exist')
            return
        sar_files = []
        for file in os.listdir(sar_dir):
            if file.endswith('.tif') and 'normalized_processed' in file:
                sar_files.append(file)
        matched_files = []
        unmatched_files = []
        for sar_file in sar_files:
            base_name = sar_file.replace('.tif', '')
            contour_folder = os.path.join(contours_dir, base_name)
            if os.path.exists(contour_folder):
                positions_path = os.path.join(contour_folder, 'contour_positions.csv')
                statistics_path = os.path.join(contour_folder, 'contour_statistics_enhanced_with_dem.csv')
                if os.path.exists(positions_path) and os.path.exists(statistics_path):
                    matched_files.append(sar_file)
                else:
                    unmatched_files.append(f'{sar_file} (missing necessary files)')
            else:
                unmatched_files.append(f'{sar_file} (no corresponding contour folder)')
        preview_text = f'📊 File Matching Check Results:\n\n'
        preview_text += f'✅ Processable files: {len(matched_files)} files\n'
        preview_text += f'❌ Unprocessable files: {len(unmatched_files)} files\n'
        preview_text += f'📁 Total SAR files: {len(sar_files)} files\n\n'
        if matched_files:
            preview_text += '✅ Processable files:\n'
            for i, file in enumerate(matched_files[:10]):
                preview_text += f'  {i + 1}. {file}\n'
            if len(matched_files) > 10:
                preview_text += f'  ... {len(matched_files) - 10} more files\n'
        if unmatched_files:
            preview_text += '\n❌ Unprocessable files:\n'
            for i, file in enumerate(unmatched_files[:5]):
                preview_text += f'  {i + 1}. {file}\n'
            if len(unmatched_files) > 5:
                preview_text += f'  ... {len(unmatched_files) - 5} more files\n'
        self.batch_preview.setText(preview_text)

    def start_batch_processing(self):
        sar_dir = self.batch_sar_dir.text().strip()
        contours_dir = self.batch_contours_dir.text().strip()
        model_path = self.batch_model_path.text().strip()
        output_dir = self.batch_output_dir.text().strip()
        if not all([sar_dir, contours_dir, model_path, output_dir]):
            QMessageBox.warning(self, 'Error', 'Please fill in all necessary paths')
            return
        if not os.path.exists(sar_dir):
            QMessageBox.warning(self, 'Error', 'SAR image folder does not exist')
            return
        if not os.path.exists(contours_dir):
            QMessageBox.warning(self, 'Error', 'Contour folder does not exist')
            return
        if not os.path.exists(model_path):
            QMessageBox.warning(self, 'Error', 'Model file does not exist')
            return
        os.makedirs(output_dir, exist_ok=True)
        config = {'sar_dir': sar_dir, 'contours_dir': contours_dir, 'model_path': model_path, 'output_dir': output_dir}
        self.batch_worker = BatchWorker(config)
        self.batch_worker.progress_updated.connect(self.batch_progress.setValue)
        self.batch_worker.log_updated.connect(self.batch_log.append)
        self.batch_worker.current_file_updated.connect(self.batch_current_file.setText)
        self.batch_worker.finished.connect(self.on_batch_finished)
        self.batch_worker.error_occurred.connect(self.on_batch_error)
        self.batch_start_btn.setEnabled(False)
        self.batch_stop_btn.setEnabled(True)
        self.batch_progress.setValue(0)
        self.batch_log.clear()
        self.batch_current_file.setText('Ready to start...')
        self.batch_worker.start()

    def stop_batch_processing(self):
        if self.batch_worker:
            self.batch_worker.stop()
            self.batch_log.append('Stopping batch processing...')

    def on_batch_finished(self):
        self.batch_start_btn.setEnabled(True)
        self.batch_stop_btn.setEnabled(False)
        self.batch_current_file.setText('Processing completed')
        self.batch_progress.setValue(100)
        QMessageBox.information(self, 'Completed', 'Batch processing completed!\n\nPlease check the result files in the output directory.')

    def on_batch_error(self, error_msg):
        self.batch_start_btn.setEnabled(True)
        self.batch_stop_btn.setEnabled(False)
        self.batch_current_file.setText('Processing error')
        self.batch_log.append(f'❌ Batch processing error: {error_msg}')
        QMessageBox.critical(self, 'Error', f'An error occurred during batch processing:\n{error_msg}')

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', required=True)
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--candidate-ids')
    parser.add_argument('--projected-mask')
    args = parser.parse_args()
    if bool(args.candidate_ids) != bool(args.projected_mask):
        parser.error('--candidate-ids and --projected-mask must be provided together')
    model, scaler, device = load_model_and_scaler(args.checkpoint, args.table)
    frame = pd.read_csv(args.table) if args.table.endswith('.csv') else pd.read_excel(args.table)
    transform = get_prediction_transform()
    rows = []
    for _, row in frame.iterrows():
        identifier = row.iloc[0]
        identifier = str(int(identifier)) if isinstance(identifier, (int, float, np.integer, np.floating)) else str(identifier)
        features = row.iloc[1:-1].to_numpy(dtype=np.float32)
        if features.shape != (17,) or not np.isfinite(features).all():
            raise ValueError('Expected 17 finite numerical features')
        label, confidence = predict_single_contour(model, str(Path(args.image_dir) / (identifier + '.png')), features, transform, scaler, device)
        rows.append([identifier, label, confidence])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=['id', 'prediction', 'confidence']).to_csv(args.output, index=False)
    if args.projected_mask:
        from saredge.projection import project
        project(args.output,args.candidate_ids,args.projected_mask)
if __name__ == '__main__':
    main()

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / '.runtime'), str(_ROOT)]
from saredge.training import seed_training, segmentation_split
from saredge.efficiency import measure_epochs
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.warp import calculate_default_transform, reproject, Resampling
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm, trange
import warnings
warnings.filterwarnings('ignore')
MODE = 'predict'

def preprocess_dem(reference_path, dem_path, output_path):
    print('=' * 60)
    print('DEM预处理')
    print('=' * 60)
    with rasterio.open(reference_path) as ref:
        ref_bounds = ref.bounds
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height
        ref_resolution = ref.res[0]
        ref_profile = ref.profile.copy()
    print(f'参考图像: {reference_path}')
    print(f'  分辨率: {ref_resolution:.4f}°')
    print(f'  范围: {ref_bounds}')
    print(f'  大小: {ref_width}x{ref_height}')
    with rasterio.open(dem_path) as dem:
        dem_bounds = dem.bounds
        dem_crs = dem.crs
    print(f'\nDEM原始: {dem_path}')
    print(f'  范围: {dem_bounds}')
    left = max(dem_bounds.left, ref_bounds.left)
    bottom = max(dem_bounds.bottom, ref_bounds.bottom)
    right = min(dem_bounds.right, ref_bounds.right)
    top = min(dem_bounds.top, ref_bounds.top)
    if left >= right or bottom >= top:
        print('错误: DEM和参考图像范围没有交集')
        return None
    print(f'\n交叉区域: left={left:.4f}, bottom={bottom:.4f}, right={right:.4f}, top={top:.4f}')
    out_transform = rasterio.transform.from_bounds(left, bottom, right, top, ref_width, ref_height)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(dem_path) as src:
        profile = ref_profile.copy()
        profile.update({'transform': out_transform, 'width': ref_width, 'height': ref_height, 'crs': ref_crs})
        print(f'\n重采样中: {ref_width}x{ref_height} (保留DEM数据，裁剪到参考图像范围)')
        with rasterio.open(output_path, 'w', **profile) as dst:
            for i in tqdm(range(1, src.count + 1), desc='重采样'):
                src_data = src.read(i)
                dst_data = np.zeros((ref_height, ref_width), dtype=src_data.dtype)
                reproject(source=src_data, destination=dst_data, src_transform=src.transform, src_crs=src.crs, dst_transform=out_transform, dst_crs=ref_crs, resampling=Resampling.bilinear)
                dst.write(dst_data, i)
    print(f'\nDEM已保存: {output_path}')
    return output_path

class TIFDataset(Dataset):

    def __init__(self, tif_path, dem_path=None, shp_path=None, tile_size=512, overlap=64, mode='train', skip_nodata=False):
        self.tif_path = tif_path
        self.dem_path = dem_path
        self.tile_size = tile_size
        self.overlap = overlap
        self.mode = mode
        self.skip_nodata = skip_nodata
        with rasterio.open(tif_path) as src:
            self.width = src.width
            self.height = src.height
            self.crs = src.crs
            self.transform = src.transform
            self.nodata = src.nodata
            self.resolution = src.res[0]
            if shp_path and os.path.exists(shp_path):
                import geopandas as gpd
                self.gdf = gpd.read_file(shp_path)
                if self.gdf.crs != self.crs:
                    self.gdf = self.gdf.to_crs(self.crs)
            else:
                self.gdf = None
        self.windows = self._generate_windows()
        print(f'生成 {len(self.windows)} 个有效tiles')

    def _generate_windows(self):
        windows = []
        step = self.tile_size if self.mode == 'train' else self.tile_size - self.overlap
        rows = list(range(0, self.height - self.tile_size + 1, step))
        cols = list(range(0, self.width - self.tile_size + 1, step))
        if rows and rows[-1] + self.tile_size < self.height:
            rows.append(self.height - self.tile_size)
        if cols and cols[-1] + self.tile_size < self.width:
            cols.append(self.width - self.tile_size)
        for row in rows:
            for col in cols:
                w = min(self.tile_size, self.width - col)
                h = min(self.tile_size, self.height - row)
                windows.append(Window(col, row, w, h))
        if self.mode == 'train':
            windows = [v for v in windows if int(v.col_off) % self.tile_size == 0 and int(v.row_off) % self.tile_size == 0]
        return windows

    def _is_invalid_tile(self, window):
        with rasterio.open(self.tif_path) as src:
            data = src.read(window=window)
        if self.nodata is not None:
            if np.all(data == self.nodata):
                return True
        elif np.all(data == 0):
            return True
        return False

    def _create_mask(self, window):
        from shapely.geometry import box
        from rasterio.features import rasterize
        bounds = rasterio.windows.bounds(window, self.transform)
        w, h = (window.width, window.height)
        mask = np.zeros((h, w), dtype=np.uint8)
        if self.gdf is not None:
            window_geom = box(*bounds)
            intersecting = self.gdf[self.gdf.intersects(window_geom)]
            if not intersecting.empty:
                shapes = [(geom, 1) for geom in intersecting.geometry]
                mask = rasterize(shapes, out_shape=(h, w), transform=rasterio.windows.transform(window, self.transform), fill=0, dtype=np.uint8)
        pad_h = self.tile_size - h
        pad_w = self.tile_size - w
        if pad_h > 0 or pad_w > 0:
            mask = np.pad(mask, [(0, pad_h), (0, pad_w)], mode='constant', constant_values=0)
        return mask

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        with rasterio.open(self.tif_path) as src:
            image = src.read(window=window)
        if self.dem_path and os.path.exists(self.dem_path):
            with rasterio.open(self.dem_path) as dem_src:
                dem_data = dem_src.read(window=window)
                image = np.concatenate([image, dem_data], axis=0)
        if len(image.shape) == 3:
            image = np.transpose(image, (1, 2, 0))
        pad_h = self.tile_size - image.shape[0]
        pad_w = self.tile_size - image.shape[1]
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, [(0, pad_h), (0, pad_w), (0, 0)], mode='reflect')
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        if self.mode == 'train':
            mask = self._create_mask(window)
            mask = torch.from_numpy(mask).long()
            return (image, mask)
        else:
            window_info = {'col': window.col_off, 'row': window.row_off}
            return (image, torch.zeros(self.tile_size, self.tile_size, dtype=torch.long), window_info)

from saredge.comparison_models_v2 import DeepLabV3Plus, model_metadata as get_model_metadata, validate_model_checkpoint, validate_dem_grid


class SegmentationTrainer:

    def __init__(self, model, device, lr=0.001, lake_weight=5.0):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0001)
        self.criterion = FocalLoss(gamma=2.0, lake_weight=lake_weight)
        self.scheduler = None

    def set_scheduler(self, total_epochs):
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=total_epochs, eta_min=1e-06)

    def train_epoch(self, dataloader, epoch, total_epochs):
        self.model.train()
        total_loss, correct, total = (0, 0, 0)
        lake_total, lake_correct = (0, 0)
        bg_total, bg_correct = (0, 0)
        pbar = tqdm(dataloader, desc=f'Train {epoch + 1}/{total_epochs}', leave=False)
        for images, masks in pbar:
            images = images.to(self.device).float()
            masks = masks.to(self.device).long()
            self.optimizer.zero_grad()
            outputs = self.model(images)
            outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += masks.nelement()
            correct += (predicted == masks).sum().item()
            lake_mask = masks == 1
            if lake_mask.sum() > 0:
                lake_total += lake_mask.sum().item()
                lake_correct += ((predicted == 1) & lake_mask).sum().item()
            bg_mask = masks == 0
            if bg_mask.sum() > 0:
                bg_total += bg_mask.sum().item()
                bg_correct += ((predicted == 0) & bg_mask).sum().item()
            pbar.set_postfix({'Loss': f'{total_loss / (pbar.n + 1):.4f}', 'Bg': f'{100 * bg_correct / bg_total:.0f}%' if bg_total > 0 else '0%', 'Lake': f'{100 * lake_correct / lake_total:.0f}%' if lake_total > 0 else '0%'})
        bg_recall = 100 * bg_correct / bg_total if bg_total > 0 else 0
        lake_recall = 100 * lake_correct / lake_total if lake_total > 0 else 0
        return (total_loss / len(dataloader), 100 * correct / total, bg_recall, lake_recall)

    def validate(self, dataloader):
        self.model.eval()
        total_loss, correct, total = (0, 0, 0)
        lake_total, lake_correct = (0, 0)
        bg_total, bg_correct = (0, 0)
        with torch.no_grad():
            for images, masks in dataloader:
                images = images.to(self.device).float()
                masks = masks.to(self.device).long()
                outputs = self.model(images)
                outputs = F.interpolate(outputs, size=masks.shape[1:], mode='bilinear', align_corners=False)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += masks.nelement()
                correct += (predicted == masks).sum().item()
                lake_mask = masks == 1
                if lake_mask.sum() > 0:
                    lake_total += lake_mask.sum().item()
                    lake_correct += ((predicted == 1) & lake_mask).sum().item()
                bg_mask = masks == 0
                if bg_mask.sum() > 0:
                    bg_total += bg_mask.sum().item()
                    bg_correct += ((predicted == 0) & bg_mask).sum().item()
        bg_recall = 100 * bg_correct / bg_total if bg_total > 0 else 0
        lake_recall = 100 * lake_correct / lake_total if lake_total > 0 else 0
        return (total_loss / len(dataloader), 100 * correct / total, bg_recall, lake_recall)

class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, lake_weight=2.0):
        super().__init__()
        self.gamma = gamma
        self.lake_weight = lake_weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        weight = torch.where(targets == 1, self.lake_weight, 1.0)
        focal_loss = weight * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def train_model(config):
    seed_training(config)
    if not config.get('shp_path') or not Path(config['shp_path']).is_file():
        raise FileNotFoundError('Training requires a reference shapefile')
    print('=' * 60)
    print('DeepLabV3+ 训练模式')
    print('=' * 60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')
    dem_path = config.get('dem_preprocessed')
    if not dem_path or not Path(dem_path).is_file():
        raise FileNotFoundError('DeepLabV3+ requires dem_preprocessed during training and prediction')
    validate_dem_grid(config['tif_path'],dem_path)
    dataset = TIFDataset(config['tif_path'], dem_path=dem_path, shp_path=config.get('shp_path'), tile_size=config['tile_size'], overlap=config['overlap'], mode='train')
    lake_area = 0
    total_area = dataset.width * dataset.height * dataset.resolution ** 2
    if dataset.gdf is not None:
        lake_area = dataset.gdf.geometry.area.sum()
    lake_weight = 10.0
    bg_weight = 1.0
    print(f'类别权重: 背景={bg_weight:.2f}, 湖泊={lake_weight:.2f}')
    train_idx, val_idx, test_idx = segmentation_split(dataset, config)
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=config['batch_size'], shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=config['batch_size'], shuffle=False, num_workers=0, pin_memory=True)
    print(f'训练tiles: {len(train_idx)}, 验证tiles: {len(val_idx)}')
    model = DeepLabV3Plus(in_channels=2, n_classes=2, pretrained=config.get('backbone_pretrained', True))
    trainer = SegmentationTrainer(model, device, lr=config['learning_rate'], lake_weight=lake_weight)
    trainer.set_scheduler(config['epochs'])
    import json
    (Path(config['model_path']).parent/'model_manifest.json').write_text(json.dumps({'model_metadata': get_model_metadata(model), 'effective_config': config},ensure_ascii=False,indent=2),encoding='utf-8')
    total_params = sum((p.numel() for p in model.parameters()))
    print(f'模型参数: {total_params / 1000000.0:.1f}M')
    best_val_loss = float('inf')
    train_history = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': [], 'val_lake_recall': []}
    for epoch in measure_epochs(trange(config['epochs'], desc='训练'),Path(config['model_path']).parent/'epoch_times.json'):
        train_loss, train_acc, bg_recall, lake_recall = trainer.train_epoch(train_loader, epoch, config['epochs'])
        val_loss, val_acc, val_bg_recall, val_lake_recall = trainer.validate(val_loader)
        print(f'Epoch {epoch + 1}: Loss={train_loss:.4f}, Acc={train_acc:.1f}% | ValLoss={val_loss:.4f}, ValAcc={val_acc:.1f}%, ValBg={val_bg_recall:.1f}%, ValLake={val_lake_recall:.1f}%')
        train_history['loss'].append(train_loss)
        train_history['acc'].append(train_acc)
        train_history['val_loss'].append(val_loss)
        train_history['val_acc'].append(val_acc)
        train_history['val_lake_recall'].append(val_lake_recall)
        (Path(config['model_path']).parent/'history.json').write_text(json.dumps(train_history,indent=2),encoding='utf-8')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model': model.state_dict(), 'config': config, 'model_metadata': get_model_metadata(model), 'train_history': train_history, 'epoch': epoch + 1}, config['model_path'].replace('.pth', '_best.pth'))
            print(f'  -> 已保存当前最优模型 (val_loss={val_loss:.4f})')
    torch.save({'model': model.state_dict(), 'config': config, 'model_metadata': get_model_metadata(model), 'train_history': train_history, 'epochs': config['epochs']}, config['model_path'])
    print(f'\n模型已保存: {config['model_path']}')
    return (model, device)

class PredictDataset(Dataset):

    def __init__(self, tif_path, dem_path=None, tile_size=512, overlap=64):
        self.tif_path = tif_path
        self.dem_path = dem_path
        self.tile_size = tile_size
        self.overlap = overlap
        with rasterio.open(tif_path) as src:
            self.height, self.width = (src.height, src.width)
            self.transform = src.transform
            self.crs = src.crs
            self.profile = src.profile.copy()
            self.nodata = src.nodata
            if dem_path and os.path.exists(dem_path):
                with rasterio.open(dem_path) as dem_src:
                    self.dem_data = dem_src.read()
                    self.dem_data = np.transpose(self.dem_data, (1, 2, 0))
            else:
                self.dem_data = None
        self.windows = self._generate_windows()

    def _generate_windows(self):
        windows = []
        step = self.tile_size - self.overlap
        rows = list(range(0, self.height - self.tile_size + 1, step))
        cols = list(range(0, self.width - self.tile_size + 1, step))
        if rows and rows[-1] + self.tile_size < self.height:
            rows.append(self.height - self.tile_size)
        if cols and cols[-1] + self.tile_size < self.width:
            cols.append(self.width - self.tile_size)
        for row in rows:
            for col in cols:
                w = min(self.tile_size, self.width - col)
                h = min(self.tile_size, self.height - row)
                windows.append((col, row, w, h))
        return windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        col, row, w, h = self.windows[idx]
        window = Window(col, row, w, h)
        with rasterio.open(self.tif_path) as src:
            image = src.read(window=window)
        if self.dem_data is not None:
            dem_tile = self.dem_data[row:row + h, col:col + w, :]
            dem_tile = np.transpose(dem_tile, (2, 0, 1))
            image = np.concatenate([image, dem_tile], axis=0)
        image = np.transpose(image, (1, 2, 0))
        pad_h = self.tile_size - image.shape[0]
        pad_w = self.tile_size - image.shape[1]
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, [(0, pad_h), (0, pad_w), (0, 0)], mode='reflect')
        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)
        return (image, col, row, w, h)

def predict(model_path, input_path, output_path, config):
    print('=' * 60)
    print('DeepLabV3+ 预测模式 (全内存 tile 分块)')
    print('=' * 60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = DeepLabV3Plus(in_channels=2, n_classes=2, pretrained=False)
    validate_model_checkpoint(checkpoint, model)
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()
    print(f'模型已加载: {model_path}')
    dem_path = config.get('dem_preprocessed')
    if not dem_path or not Path(dem_path).is_file():
        raise FileNotFoundError('DeepLabV3+ requires dem_preprocessed during training and prediction')
    validate_dem_grid(input_path,dem_path)
    tile_size = config['tile_size']
    overlap = config['overlap']
    batch_size = config.get('predict_batch_size', 64)
    print('阶段0：加载图像到内存...')
    import time, gc
    t0 = time.time()
    with rasterio.open(input_path) as src:
        height, width = (src.height, src.width)
        transform, crs = (src.transform, src.crs)
        profile = src.profile.copy()
        image_data = src.read().astype(np.uint8)
        print(f'  图像: {image_data.shape}, {image_data.dtype}, {time.time() - t0:.1f}s')
    if os.path.exists(dem_path):
        with rasterio.open(dem_path) as dem_src:
            dem_raw = dem_src.read()[0].astype(np.float16)
        print(f'  DEM: {dem_raw.shape}, {time.time() - t0:.1f}s')
    else:
        dem_raw = None
    windows = []
    step = tile_size - overlap
    rows = list(range(0, height - tile_size + 1, step))
    cols = list(range(0, width - tile_size + 1, step))
    if rows and rows[-1] + tile_size < height:
        rows.append(height - tile_size)
    if cols and cols[-1] + tile_size < width:
        cols.append(width - tile_size)
    for row in rows:
        for col in cols:
            w = min(tile_size, width - col)
            h = min(tile_size, height - row)
            windows.append((col, row, w, h))
    n_tiles = len(windows)
    print(f'  窗口数: {n_tiles}, tile: {tile_size}, overlap: {overlap}, batch_size: {batch_size}')
    print(f'  加载总耗时: {time.time() - t0:.1f}s')
    print('阶段1：推理...')
    sum_prob = np.zeros((height, width), dtype=np.float16)
    sum_count = np.zeros((height, width), dtype=np.float16)
    n_batches = (n_tiles + batch_size - 1) // batch_size
    with torch.no_grad():
        for batch_i in tqdm(range(n_batches), desc='推理'):
            start = batch_i * batch_size
            end = min(start + batch_size, n_tiles)
            batch_windows = windows[start:end]
            B = end - start
            tiles = []
            for col, row, w, h in batch_windows:
                img_tile = image_data[:, row:row + h, col:col + w]
                if dem_raw is not None:
                    dem_tile = dem_raw[row:row + h, col:col + w]
                    dem_tile = dem_tile[np.newaxis, :, :]
                    img_tile = np.concatenate([img_tile, dem_tile], axis=0)
                img_tile = np.transpose(img_tile, (1, 2, 0))
                pad_h = tile_size - h
                pad_w = tile_size - w
                if pad_h or pad_w:
                    img_tile = np.pad(img_tile, [(0, pad_h), (0, pad_w), (0, 0)], mode='reflect')
                tiles.append(img_tile)
            images = torch.from_numpy(np.stack(tiles).astype(np.float32))
            images = images.permute(0, 3, 1, 2).to(device)
            outputs = model(images)
            if outputs.shape[-1] != tile_size:
                outputs = F.interpolate(outputs, size=(tile_size, tile_size), mode='bilinear', align_corners=False)
            lake_probs = F.softmax(outputs, dim=1)[:, 1, :, :].cpu().numpy().astype(np.float16)
            for i, (col, row, w, h) in enumerate(batch_windows):
                sum_prob[row:row + h, col:col + w] += lake_probs[i][:h, :w]
                sum_count[row:row + h, col:col + w] += np.float16(1)
            del tiles, images, outputs, lake_probs
    print('阶段2：写结果...')
    threshold = config.get('predict_threshold', 0.5)
    profile_cls = profile.copy()
    profile_cls.update(driver='GTiff', count=1, dtype=rasterio.uint8, tiled=True, blockxsize=tile_size, blockysize=tile_size, compress='deflate', bigtiff='YES')
    prob_out_path = output_path.replace('.tif', '_prob.tif')
    profile_prb = profile.copy()
    profile_prb.update(driver='GTiff', count=1, dtype=rasterio.float32, tiled=True, blockxsize=tile_size, blockysize=tile_size, compress='deflate', bigtiff='YES')
    with rasterio.open(output_path, 'w', **profile_cls) as dst_cls, rasterio.open(prob_out_path, 'w', **profile_prb) as dst_prb:
        for col, row, w, h in tqdm(windows, desc='写结果'):
            sp = sum_prob[row:row + h, col:col + w]
            sc = sum_count[row:row + h, col:col + w]
            avg = (sp / np.maximum(sc, 1)).astype(np.float32)
            cls = (avg >= threshold).astype(np.uint8)
            win = Window(col, row, w, h)
            dst_cls.write(cls[np.newaxis], window=win)
            dst_prb.write(avg[np.newaxis], window=win)
    del sum_prob, sum_count
    gc.collect()
    print(f'\n完成！')
    print(f'  类别图: {output_path}  (0=背景, 1=湖泊)')
    print(f'  概率图: {prob_out_path}  (float32, 0~1)')

def main():
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8'))
    key = 'model_path' if 'model_path' in config else 'model_save_path'
    Path(config[key]).parent.mkdir(parents=True, exist_ok=True)
    train_model(config)
if __name__ == '__main__':
    main()

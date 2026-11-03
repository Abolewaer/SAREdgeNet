import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / '.runtime'), str(_ROOT)]
from saredge.training import seed_training, segmentation_split
from saredge.efficiency import measure_epochs
import os
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
import geopandas as gpd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

class _SARDataset(Dataset):

    def __init__(self, tif_path, shp_path, tile_size=512, overlap=64, transform=None, mode='train'):
        self.tif_path = tif_path
        self.shp_path = shp_path
        self.tile_size = tile_size
        self.overlap = overlap
        self.transform = transform
        self.mode = mode
        self._src = None
        with rasterio.open(tif_path) as src:
            self.width = src.width
            self.height = src.height
            self.crs = src.crs
            self.transform_matrix = src.transform
            self.bounds = src.bounds
        if shp_path and os.path.exists(shp_path):
            self.gdf = gpd.read_file(shp_path)
            if self.gdf.crs != self.crs:
                self.gdf = self.gdf.to_crs(self.crs)
        else:
            self.gdf = None
        self.full_mask = self._build_full_mask()
        self.windows = self._generate_windows()
        self.positive_ratios = self._compute_positive_ratios() if self.mode == 'train' else None

    def _build_full_mask(self):
        if self.gdf is None or self.gdf.empty:
            return np.zeros((self.height, self.width), dtype=np.uint8)
        print('🗺️ 预栅格化整幅标签掩码...')
        shapes = ((geom, 1) for geom in self.gdf.geometry if geom is not None and (not geom.is_empty))
        return rasterize(shapes=shapes, out_shape=(self.height, self.width), transform=self.transform_matrix, fill=0, dtype=np.uint8)

    def _get_raster_source(self):
        if self._src is None:
            self._src = rasterio.open(self.tif_path)
        return self._src

    def _slice_window_array(self, array, window):
        row_start = int(window.row_off)
        row_end = row_start + int(window.height)
        col_start = int(window.col_off)
        col_end = col_start + int(window.width)
        return array[row_start:row_end, col_start:col_end]

    def _get_mask_for_window(self, window):
        return self._slice_window_array(self.full_mask, window).copy()

    def _generate_windows(self):
        windows = []
        step = self.tile_size if self.mode == 'train' else self.tile_size - self.overlap
        rows = list(range(0, self.height - self.tile_size + 1, step))
        cols = list(range(0, self.width - self.tile_size + 1, step))
        total_windows = len(rows) * len(cols)
        print(f'生成分块窗口: {len(rows)} x {len(cols)} = {total_windows} 个')
        with tqdm(total=total_windows, desc='生成窗口', unit='个') as pbar:
            for row in rows:
                for col in cols:
                    windows.append(Window(col, row, self.tile_size, self.tile_size))
                    pbar.update(1)
        if self.mode == 'train':
            windows = [v for v in windows if int(v.col_off) % self.tile_size == 0 and int(v.row_off) % self.tile_size == 0]
        return windows

    def _compute_positive_ratios(self):
        if self.gdf is None:
            return [0.0] * len(self.windows)
        positive_ratios = []
        print('📌 统计每个分块中的目标像素占比...')
        with tqdm(total=len(self.windows), desc='统计正样本分块', unit='个') as pbar:
            for window in self.windows:
                mask = self._get_mask_for_window(window)
                positive_ratios.append(float(mask.mean()))
                pbar.update(1)
        positive_tiles = sum((r > 0 for r in positive_ratios))
        print(f'   ✓ 含目标分块: {positive_tiles:,}/{len(self.windows):,} ({positive_tiles / len(self.windows) * 100:.2f}%)')
        return positive_ratios

    def __len__(self):
        return len(self.windows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_src'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._src = None

    def __getitem__(self, idx):
        window = self.windows[idx]
        src = self._get_raster_source()
        image = src.read(window=window)
        mask = self._get_mask_for_window(window)
        if len(image.shape) == 3:
            image = np.transpose(image, (1, 2, 0))
        else:
            image = np.expand_dims(image, axis=-1)
        image = image.astype(np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()
        if self.transform:
            image = self.transform(image)
        if self.mode == 'train':
            return (image, mask)
        window_info = {'col': window.col_off, 'row': window.row_off, 'width': window.width, 'height': window.height}
        return (image, mask, window_info)

from saredge.sar_dem_baselines import PSPNet, SarDemDatasetMixin, checkpoint_metadata, validate_checkpoint


class TIFDataset(SarDemDatasetMixin, _SARDataset):
    pass


class TverskyLoss(nn.Module):

    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-06):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        targets_oh = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        tp = (inputs * targets_oh).sum(dim=(2, 3))
        fp = (inputs * (1 - targets_oh)).sum(dim=(2, 3))
        fn = ((1 - inputs) * targets_oh).sum(dim=(2, 3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1 - tversky.mean()

class FocalTverskyLoss(nn.Module):

    def __init__(self, alpha=0.7, beta=0.3, gamma=4.0, smooth=1e-06):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        targets_oh = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()
        tp = (inputs * targets_oh).sum(dim=(2, 3))
        fp = (inputs * (1 - targets_oh)).sum(dim=(2, 3))
        fn = ((1 - inputs) * targets_oh).sum(dim=(2, 3))
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma
        return focal_tversky.mean()

class SegmentationLoss(nn.Module):

    def __init__(self, tversky_w=0.4, focal_tversky_w=0.3, ce_w=0.3, alpha=0.7, beta=0.3, gamma=4.0, ce_weight=None):
        super().__init__()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)
        self.focal_tversky = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma)
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.tversky_w = tversky_w
        self.focal_tversky_w = focal_tversky_w
        self.ce_w = ce_w

    def forward(self, inputs, targets):
        return self.tversky_w * self.tversky(inputs, targets) + self.focal_tversky_w * self.focal_tversky(inputs, targets) + self.ce_w * self.ce(inputs, targets)

class SegmentationTrainer:

    def __init__(self, model, device, learning_rate=0.001, weight_decay=0.0):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        ce_weight = torch.tensor([1.0, 30.0], device=device)
        self.criterion = SegmentationLoss(tversky_w=0.4, focal_tversky_w=0.3, ce_w=0.3, alpha=0.8, beta=0.3, gamma=4.0, ce_weight=ce_weight)

    def train_epoch(self, dataloader, epoch, total_epochs):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        class_correct = [0, 0]
        class_total = [0, 0]
        pbar = tqdm(dataloader, desc=f'训练 Epoch {epoch + 1}/{total_epochs}', leave=False, unit='batch')
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = (images.to(self.device), masks.to(self.device))
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += masks.nelement()
            correct += (predicted == masks).sum().item()
            for c in range(2):
                mask_c = masks == c
                class_total[c] += mask_c.sum().item()
                class_correct[c] += ((predicted == masks) & mask_c).sum().item()
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100 * correct / total
            acc0 = 100 * class_correct[0] / class_total[0] if class_total[0] > 0 else 0
            acc1 = 100 * class_correct[1] / class_total[1] if class_total[1] > 0 else 0
            pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Acc': f'{accuracy:.2f}%', 'Acc0': f'{acc0:.1f}%', 'Acc1': f'{acc1:.1f}%', 'GPU': f'{torch.cuda.memory_allocated() / 1024 ** 3:.1f}GB' if torch.cuda.is_available() else 'CPU'})
        return total_loss / len(dataloader)

    def validate(self, dataloader, epoch, total_epochs):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        class_correct = [0, 0]
        class_total = [0, 0]
        pbar = tqdm(dataloader, desc=f'验证 Epoch {epoch + 1}/{total_epochs}', leave=False, unit='batch')
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(pbar):
                images, masks = (images.to(self.device), masks.to(self.device))
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += masks.nelement()
                correct += (predicted == masks).sum().item()
                for c in range(2):
                    mask_c = masks == c
                    class_total[c] += mask_c.sum().item()
                    class_correct[c] += ((predicted == masks) & mask_c).sum().item()
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = 100 * correct / total
                acc0 = 100 * class_correct[0] / class_total[0] if class_total[0] > 0 else 0
                acc1 = 100 * class_correct[1] / class_total[1] if class_total[1] > 0 else 0
                pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Acc': f'{accuracy:.2f}%', 'Acc0': f'{acc0:.1f}%', 'Acc1': f'{acc1:.1f}%'})
        accuracy = 100 * correct / total
        acc0 = 100 * class_correct[0] / class_total[0] if class_total[0] > 0 else 0
        acc1 = 100 * class_correct[1] / class_total[1] if class_total[1] > 0 else 0
        return (total_loss / len(dataloader), accuracy, acc0, acc1)

def validate_config(config):
    print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
    required_paths = ['tif_path', 'model_save_path', 'output_path']
    for path_key in required_paths:
        if path_key not in config:
            raise ValueError(f'配置中缺少 {path_key}')
    if not os.path.exists(config['tif_path']):
        raise FileNotFoundError(f'TIF文件不存在: {config['tif_path']}')
    if config.get('shp_path') and (not os.path.exists(config['shp_path'])):
        print(f'⚠️ 警告: SHP文件不存在: {config['shp_path']}，将使用空掩码进行训练')
        config['shp_path'] = None
    for path_key in ['model_save_path', 'output_path']:
        output_dir = os.path.dirname(config[path_key])
        if output_dir and (not os.path.exists(output_dir)):
            os.makedirs(output_dir)
            print(f'📁 创建输出目录: {output_dir}')
    small_sample_ratio = config.get('small_sample_ratio', 0.2)
    if small_sample_ratio <= 0 or small_sample_ratio >= 1:
        raise ValueError('small_sample_ratio 必须在 (0, 1) 范围内')
    val_ratio_in_small = config.get('val_ratio_in_small', 0.25)
    if val_ratio_in_small <= 0 or val_ratio_in_small >= 1:
        raise ValueError('val_ratio_in_small 必须在 (0, 1) 范围内')
    print('✅ PSPNet配置验证通过')

def setup_environment():
    print('💻 环境信息:')
    print(f'   PyTorch版本: {torch.__version__}')
    print(f'   CUDA可用: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   CUDA版本: {torch.version.cuda}')
        print(f'   GPU数量: {torch.cuda.device_count()}')
        print(f'   当前GPU: {torch.cuda.get_device_name()}')
        print(f'   显存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f}GB')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device

def train_model(config):
    seed_training(config)
    if not config.get('shp_path') or not Path(config['shp_path']).is_file():
        raise FileNotFoundError('Training requires an existing reference shapefile')
    if not config.get('dem_path') or not Path(config['dem_path']).is_file():
        raise FileNotFoundError('Training requires dem_path')
    print('🚀 开始PSPNet模型训练')
    print('=' * 60)
    tif_path = config['tif_path']
    shp_path = config.get('shp_path')
    model_save_path = config['model_save_path']
    epochs = config['epochs']
    batch_size = config['batch_size']
    tile_size = config['tile_size']
    learning_rate = config['learning_rate']
    overlap = config['overlap']
    print(f'📋 训练配置:')
    print(f'   - TIF文件: {tif_path}')
    print(f'   - SHP文件: {shp_path or '无 (使用空掩码)'}')
    print(f'   - 分块大小: {tile_size}x{tile_size}')
    print(f'   - 重叠区域: {overlap}')
    print(f'   - 批次大小: {batch_size}')
    print(f'   - 学习率: {learning_rate}')
    print(f'   - 训练轮数: {epochs}')
    print(f'   - 模型保存路径: {model_save_path}')
    print()
    print('📊 准备数据集...')
    start_time = time.time()
    dataset = TIFDataset(tif_path, shp_path, tile_size=tile_size, overlap=overlap, mode='train', dem_path=config.get('dem_path'))
    dataset_time = time.time() - start_time
    print(f'   ✓ 数据集准备完成，用时: {dataset_time:.1f}秒')
    print(f'   ✓ 总分块数: {len(dataset):,}')
    sample_image, _ = dataset[0]
    in_channels = sample_image.shape[0]
    print(f'   ✓ 输入通道数: {in_channels}')
    small_sample_ratio = config.get('small_sample_ratio', 0.2)
    val_ratio_in_small = config.get('val_ratio_in_small', 0.25)
    train_indices, val_indices, large_test_indices = segmentation_split(dataset, config)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    large_test_dataset = torch.utils.data.Subset(dataset, large_test_indices)
    train_positive_ratios = [dataset.positive_ratios[idx] for idx in train_indices]
    train_sample_weights = [1.0 + min(r * 200.0, 29.0) for r in train_positive_ratios]
    train_sampler = WeightedRandomSampler(weights=torch.DoubleTensor(train_sample_weights), num_samples=len(train_sample_weights), replacement=True)
    configured_num_workers = config['num_workers']
    loader_num_workers = 0 if os.name == 'nt' else configured_num_workers
    if loader_num_workers != configured_num_workers:
        print(f'⚠️ Windows 环境下 num_workers 已从 {configured_num_workers} 调整为 {loader_num_workers}')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=loader_num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=loader_num_workers, pin_memory=True)
    large_test_loader = DataLoader(large_test_dataset, batch_size=batch_size, shuffle=False, num_workers=loader_num_workers, pin_memory=True)
    print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
    print(f'       📊 训练集: {len(train_dataset):,} 个分块 ({len(train_dataset) / len(dataset) * 100:.1f}%)')
    print(f'       📊 验证集: {len(val_dataset):,} 个分块 ({len(val_dataset) / len(dataset) * 100:.1f}%)')
    print(f'       🎯 大规模测试集: {len(large_test_dataset):,} 个分块 ({len(large_test_dataset) / len(dataset) * 100:.1f}%)')
    print(f'   ✓ 训练批次数: {len(train_loader):,}')
    print(f'   ✓ 验证批次数: {len(val_loader):,}')
    print(f'   ✓ 大测试批次数: {len(large_test_loader):,}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'💻 使用设备: {device}')
    model = PSPNet(in_channels=in_channels, n_classes=2)
    trainer = SegmentationTrainer(model, device, learning_rate=learning_rate, weight_decay=config.get('weight_decay', 0.0))
    import json
    manifest = {'model_metadata': checkpoint_metadata(model), 'effective_config': config, 'optimizer': {'name': 'Adam', 'learning_rate': learning_rate, 'weight_decay': config.get('weight_decay', 0.0)}, 'model_selection': 'validation pixel accuracy; first maximum', 'experiment_version': 'SARDEM_v2'}
    (Path(model_save_path).parent/'model_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    total_params = sum((p.numel() for p in model.parameters()))
    trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    print(f'🧠 模型信息 (PSPNet):')
    print(f'   ✓ 总参数数: {total_params:,}')
    print(f'   ✓ 可训练参数: {trainable_params:,}')
    print('\n' + '=' * 60)
    print('🎯 开始训练')
    print('=' * 60)
    best_accuracy = float('-inf')
    train_losses = []
    val_losses = []
    val_accuracies = []
    epoch_pbar = trange(epochs, desc='总进度', unit='epoch', position=0)
    for epoch in measure_epochs(epoch_pbar,Path(model_save_path).parent/'epoch_times.json'):
        epoch_start_time = time.time()
        train_loss = trainer.train_epoch(train_loader, epoch, epochs)
        train_losses.append(train_loss)
        val_loss, val_accuracy, val_acc0, val_acc1 = trainer.validate(val_loader, epoch, epochs)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        (Path(model_save_path).parent/'history.json').write_text(json.dumps({'train_loss': train_losses, 'val_loss': val_losses, 'val_accuracy': val_accuracies},indent=2),encoding='utf-8')
        epoch_time = time.time() - epoch_start_time
        epoch_pbar.set_postfix({'Train Loss': f'{train_loss:.4f}', 'Val Loss': f'{val_loss:.4f}', 'Val Acc': f'{val_accuracy:.2f}%', 'Time': f'{epoch_time:.1f}s'})
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': trainer.optimizer.state_dict(), 'train_loss': train_loss, 'val_loss': val_loss, 'val_accuracy': val_accuracy, 'val_acc0': val_acc0, 'val_acc1': val_acc1, 'config': config, 'model_metadata': checkpoint_metadata(model)}, model_save_path)
            tqdm.write(f'🎉 保存最佳模型 (Epoch {epoch + 1}): 总Acc {val_accuracy:.2f}% | Acc0 {val_acc0:.1f}% | Acc1 {val_acc1:.1f}%')
        if config.get('early_stopping', False) and len(val_accuracies) - 1 - int(np.argmax(val_accuracies)) >= config.get('patience', 10):
            break
    print(f'   ✓ 模型已保存: {model_save_path}')
    print('\n' + '=' * 60)
    print('🎯 大规模测试集评估')
    print('=' * 60)
    checkpoint = torch.load(model_save_path, map_location=device, weights_only=False)
    validate_checkpoint(checkpoint, model)
    model.load_state_dict(checkpoint['model_state_dict'])
    large_test_loss, large_test_accuracy, large_test_acc0, large_test_acc1 = trainer.validate(large_test_loader, 0, 1)
    print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
    print(f'   ✅ 训练集大小: {len(train_dataset):,} 个分块')
    print(f'   ✅ 验证集总精度: {best_accuracy:.2f}%  (Acc0={val_acc0:.1f}% | Acc1={val_acc1:.1f}%)')
    print(f'   📊 大测试集总精度: {large_test_accuracy:.2f}%  (Acc0={large_test_acc0:.1f}% | Acc1={large_test_acc1:.1f}%)')
    dataset.close()
    return (model, train_losses, val_losses, val_accuracies, large_test_loss, large_test_accuracy, large_test_acc0, large_test_acc1)

def predict_large_tif(model_path, input_tif_path, output_tif_path, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'💻 使用设备: {device}')
    with rasterio.open(input_tif_path) as src:
        width = src.width
        height = src.height
        transform = src.transform
        crs = src.crs
        in_channels = 2
    print(f'📖 读取输入文件信息: {width} x {height}, 通道数: {in_channels}')
    model = PSPNet(in_channels=in_channels, n_classes=2)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    validate_checkpoint(checkpoint, model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f'🧠 模型已加载: {model_path}')
    tile_size = config.get('tile_size', 512)
    overlap = config.get('overlap', 64)
    predict_dataset = TIFDataset(input_tif_path, None, tile_size=tile_size, overlap=overlap, mode='predict', dem_path=config.get('dem_path'))
    predict_loader = DataLoader(predict_dataset, batch_size=1, shuffle=False, num_workers=0)
    prediction = np.zeros((height, width), dtype=np.uint8)
    confidence = np.zeros((height, width), dtype=np.float32)
    print('🎯 开始分块预测...')
    start_time = time.time()
    with torch.no_grad():
        pbar = tqdm(predict_loader, desc='预测进度', unit='batch')
        for _, (images, _, window_infos) in enumerate(pbar):
            images = images.to(device)
            output = model(images)
            prob = F.softmax(output, dim=1)
            conf = prob.max(dim=1)[0].cpu().numpy()[0]
            pred = torch.argmax(output, dim=1).cpu().numpy()[0].astype(np.uint8)
            window_info = {k: v[0].item() if torch.is_tensor(v) else v[0] for k, v in window_infos.items()}
            row = window_info['row']
            col = window_info['col']
            h = window_info['height']
            w = window_info['width']
            pred = pred[:h, :w]
            conf = conf[:h, :w]
            prediction[row:row + h, col:col + w] = pred
            confidence[row:row + h, col:col + w] = conf
            pbar.update(1)
    print(f'   ✓ 预测完成，用时: {time.time() - start_time:.1f}秒')
    total_pixels = prediction.size
    class1_pixels = np.sum(prediction == 1)
    print(f'\n📈 预测统计:')
    print(f'   ✓ 总像素数: {total_pixels:,}')
    print(f'   ✓ 红色类别像素数: {class1_pixels:,} ({class1_pixels / total_pixels:.2%})')
    print('💾 保存分类结果...')
    with rasterio.open(output_tif_path, 'w', driver='GTiff', height=height, width=width, count=1, dtype=np.uint8, crs=crs, transform=transform, nodata=0, compress='lzw') as dst:
        dst.write(prediction, 1)
    print(f'   ✅ 分类结果已保存: {output_tif_path}')
    confidence_path = output_tif_path.replace('.tif', '_confidence.tif')
    with rasterio.open(confidence_path, 'w', driver='GTiff', height=height, width=width, count=1, dtype=np.float32, crs=crs, transform=transform, compress='lzw') as dst:
        dst.write(confidence, 1)
    print(f'   ✅ 置信度图已保存: {confidence_path}')
    predict_dataset.close()
    return (prediction, confidence)

def save_training_plots(train_losses, val_losses, val_accuracies, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(train_losses) + 1)
    axes[0].plot(list(epochs_range), train_losses, 'b-o', label='训练损失', linewidth=2)
    axes[0].plot(list(epochs_range), val_losses, 'r-s', label='验证损失', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('训练与验证损失曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(list(epochs_range), val_accuracies, 'g-^', linewidth=2, markersize=8)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('准确率 (%)')
    axes[1].set_title('验证集准确率曲线')
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'   ✅ 训练曲线已保存: {save_path}')

def print_final_report(train_losses, val_losses, val_accuracies, large_test_loss, large_test_accuracy, large_test_acc0, large_test_acc1, prediction, confidence, config):
    print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
    print('=' * 80)
    final_val_accuracy = max(val_accuracies) if val_accuracies else 0
    total_pixels = prediction.size
    class1_pixels = np.sum(prediction == 1)
    print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
    print(f'   📈 性能表现:')
    print(f'       - 最佳验证精度: {final_val_accuracy:.2f}%')
    print(f'       - 大规模测试精度: {large_test_accuracy:.2f}% (Acc0={large_test_acc0:.1f}% | Acc1={large_test_acc1:.1f}%)')
    print(f'       - 训练轮数: {len(train_losses)}')
    print(f'       - 最终训练损失: {train_losses[-1]:.4f}')
    print(f'       - 最终验证损失: {val_losses[-1]:.4f}')
    print(f'\n🔮 大规模预测结果分析:')
    print(f'   📊 总像素数: {total_pixels:,}')
    print(f'   📊 检测到的目标像素: {class1_pixels:,} ({class1_pixels / total_pixels:.2%})')
    if confidence is not None:
        avg_conf = np.mean(confidence)
        high_conf_pixels = np.sum(confidence > 0.8)
        print(f'   📊 平均预测置信度: {avg_conf:.3f}')
        print(f'   📊 高置信度像素 (>0.8): {high_conf_pixels:,} ({high_conf_pixels / total_pixels:.2%})')
    print(f'\n📁 输出文件:')
    print(f'   ✅ 模型文件: {config['model_save_path']}')
    print(f'   ✅ 分类结果: {config['output_path']}')

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

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
                    window = Window(col, row, self.tile_size, self.tile_size)
                    windows.append(window)
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

from saredge.sar_dem_baselines import FPN, SarDemDatasetMixin, checkpoint_metadata, validate_checkpoint


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
    if config['batch_size'] <= 0:
        raise ValueError('batch_size 必须大于 0')
    if config['epochs'] <= 0:
        raise ValueError('epochs 必须大于 0')
    if config['tile_size'] <= 0:
        raise ValueError('tile_size 必须大于 0')
    train_ratio = small_sample_ratio * (1 - val_ratio_in_small)
    val_ratio = small_sample_ratio * val_ratio_in_small
    test_ratio = 1 - small_sample_ratio
    print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
    print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
    print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
    print(f'   大测试集: {test_ratio:.1%}')
    print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')

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
    print('🚀 开始FPN模型训练')
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
    print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
    print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
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
        print(f'⚠️ Windows 环境下为避免 DataLoader 多进程序列化问题，num_workers 已从 {configured_num_workers} 调整为 {loader_num_workers}')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=loader_num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=loader_num_workers, pin_memory=True)
    large_test_loader = DataLoader(large_test_dataset, batch_size=batch_size, shuffle=False, num_workers=loader_num_workers, pin_memory=True)
    print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
    print(f'       📊 训练集: {len(train_dataset):,} 个分块 ({len(train_dataset) / len(dataset) * 100:.1f}%)')
    print(f'       📊 验证集: {len(val_dataset):,} 个分块 ({len(val_dataset) / len(dataset) * 100:.1f}%)')
    print(f'       🎯 大规模测试集: {len(large_test_dataset):,} 个分块 ({len(large_test_dataset) / len(dataset) * 100:.1f}%)')
    print(f'   ✓ 训练批次数: {len(train_loader):,}')
    oversampled_positive_tiles = sum((r > 0 for r in train_positive_ratios))
    print(f'   ✓ 训练集含目标分块: {oversampled_positive_tiles:,}/{len(train_dataset):,}')
    print(f'   ✓ 训练采样权重范围: {min(train_sample_weights):.2f} - {max(train_sample_weights):.2f}')
    print(f'   ✓ 验证批次数: {len(val_loader):,}')
    print(f'   ✓ 大测试批次数: {len(large_test_loader):,}')
    print(f'\n   🔬 实验意义:')
    print(f'       - 用{len(train_dataset):,}个样本训练模型')
    print(f'       - 在{len(large_test_dataset):,}个样本上测试泛化能力')
    print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'💻 使用设备: {device}')
    model = FPN(in_channels=in_channels, n_classes=2)
    trainer = SegmentationTrainer(model, device, learning_rate=learning_rate, weight_decay=config.get('weight_decay', 0.0))
    pretrain_path = config.get('pretrain_path')
    if pretrain_path and not Path(pretrain_path).is_file():
        raise FileNotFoundError(pretrain_path)
    if pretrain_path:
        print(f'🌱 加载预训练权重: {pretrain_path}')
        pretrain_checkpoint = torch.load(pretrain_path, map_location=device, weights_only=False)
        validate_checkpoint(pretrain_checkpoint, model)
        model.load_state_dict(pretrain_checkpoint['model_state_dict'])
        print(f'   ✓ 已加载预训练模型 (epoch={pretrain_checkpoint.get('epoch', '?')}, val_acc={pretrain_checkpoint.get('val_accuracy', 0):.2f}%)')
    import json
    manifest = {'model_metadata': checkpoint_metadata(model), 'effective_config': config, 'optimizer': {'name': 'Adam', 'learning_rate': learning_rate, 'weight_decay': config.get('weight_decay', 0.0)}, 'model_selection': 'validation pixel accuracy; first maximum', 'experiment_version': 'SARDEM_v2'}
    (Path(model_save_path).parent/'model_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    total_params = sum((p.numel() for p in model.parameters()))
    trainable_params = sum((p.numel() for p in model.parameters() if p.requires_grad))
    print(f'🧠 模型信息:')
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
    epoch_pbar.close()
    print(f'\n✅ 训练完成!')
    print(f'   ✓ 最佳验证精度: {best_accuracy:.2f}%')
    print(f'   ✓ 模型已保存: {model_save_path}')
    print('\n' + '=' * 60)
    print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
    print('=' * 60)
    checkpoint = torch.load(model_save_path, map_location=device, weights_only=False)
    validate_checkpoint(checkpoint, model)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'   🔬 正在评估 {len(large_test_dataset):,} 个测试样本...')
    large_test_loss, large_test_accuracy, large_test_acc0, large_test_acc1 = trainer.validate(large_test_loader, 0, 1)
    print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
    print(f'   ✅ 训练集大小: {len(train_dataset):,} 个分块')
    print(f'   ✅ 验证集大小: {len(val_dataset):,} 个分块')
    print(f'   ✅ 大测试集大小: {len(large_test_dataset):,} 个分块')
    print(f'   📊 验证集总精度: {best_accuracy:.2f}%  (Acc0={val_acc0:.1f}% | Acc1={val_acc1:.1f}%)')
    print(f'   📊 大测试集总精度: {large_test_accuracy:.2f}%  (Acc0={large_test_acc0:.1f}% | Acc1={large_test_acc1:.1f}%)')
    print(f'   📊 大测试集损失: {large_test_loss:.4f}')
    print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
    val_test_diff = abs(best_accuracy - large_test_accuracy)
    sample_efficiency = len(train_dataset) / len(dataset)
    print(f'   📊 样本效率: 用{sample_efficiency:.1%}的数据训练')
    print(f'   📊 验证-测试差异: {val_test_diff:.2f}%')
    if large_test_accuracy > 80:
        if val_test_diff < 3.0:
            print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
        elif val_test_diff < 8.0:
            print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
        else:
            print(f'   ⚠️ 存在一定过拟合，但大测试集表现尚可')
    elif large_test_accuracy > 70:
        print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
    else:
        print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
    checkpoint['large_test_loss'] = large_test_loss
    checkpoint['large_test_accuracy'] = large_test_accuracy
    checkpoint['sample_efficiency'] = sample_efficiency
    checkpoint['train_sample_count'] = len(train_dataset)
    checkpoint['large_test_sample_count'] = len(large_test_dataset)
    torch.save(checkpoint, model_save_path)
    dataset.close()
    return (model, train_losses, val_losses, val_accuracies, large_test_loss, large_test_accuracy, large_test_acc0, large_test_acc1)

def predict_large_tif(model_path, input_tif_path, output_tif_path, config):
    print('🔮 开始大型TIF文件预测')
    print('=' * 60)
    tile_size = config['tile_size']
    overlap = config['overlap']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'💻 使用设备: {device}')
    print('📖 读取输入文件信息...')
    with rasterio.open(input_tif_path) as src:
        width = src.width
        height = src.height
        transform = src.transform
        crs = src.crs
        in_channels = 2
    print(f'   ✓ 图像尺寸: {width} x {height}')
    print(f'   ✓ 通道数: {in_channels}')
    print(f'   ✓ 坐标系: {crs}')
    print('🧠 加载模型...')
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = FPN(in_channels=in_channels, n_classes=2)
    validate_checkpoint(checkpoint, model)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'   ✓ 加载checkpoint (Epoch {checkpoint.get('epoch', '?')})')
        if 'val_accuracy' in checkpoint:
            print(f'   ✓ 模型验证精度: {checkpoint['val_accuracy']:.2f}%')
    else:
        model.load_state_dict(checkpoint)
        print('   ✓ 加载模型权重')
    model.to(device)
    model.eval()
    print('📊 创建预测数据集...')
    predict_dataset = TIFDataset(input_tif_path, None, tile_size=tile_size, overlap=overlap, mode='predict', dem_path=config.get('dem_path'))
    predict_loader = DataLoader(predict_dataset, batch_size=1, shuffle=False, num_workers=0)
    print('📊 初始化输出数组...')
    prediction = np.zeros((height, width), dtype=np.uint8)
    confidence = np.zeros((height, width), dtype=np.float32)
    print('🎯 开始分块预测...')
    start_time = time.time()
    with torch.no_grad():
        pbar = tqdm(predict_loader, desc='预测进度', unit='batch')
        for batch_idx, (images, _, window_infos) in enumerate(pbar):
            try:
                images = images.to(device)
                output = model(images)
                prob = torch.softmax(output, dim=1)
                pred = torch.argmax(output, dim=1).cpu().numpy()
                conf = torch.max(prob, dim=1)[0].cpu().numpy()
                for i in range(images.size(0)):
                    window_info = {k: v[i].item() if torch.is_tensor(v) else v[i] for k, v in window_infos.items()}
                    row = window_info['row']
                    col = window_info['col']
                    pred_single = pred[i]
                    conf_single = conf[i]
                    if overlap > 0 and (row > 0 or col > 0):
                        start_r = overlap // 2 if row > 0 else 0
                        end_r = tile_size - overlap // 2 if row + tile_size < height else tile_size
                        start_c = overlap // 2 if col > 0 else 0
                        end_c = tile_size - overlap // 2 if col + tile_size < width else tile_size
                        pred_single = pred_single[start_r:end_r, start_c:end_c]
                        conf_single = conf_single[start_r:end_r, start_c:end_c]
                        r_start, r_end = (row + start_r, row + end_r)
                        c_start, c_end = (col + start_c, col + end_c)
                    else:
                        r_start, r_end = (row, row + tile_size)
                        c_start, c_end = (col, col + tile_size)
                    r_end = min(r_end, height)
                    c_end = min(c_end, width)
                    prediction[r_start:r_end, c_start:c_end] = pred_single[:r_end - r_start, :c_end - c_start]
                    confidence[r_start:r_end, c_start:c_end] = conf_single[:r_end - r_start, :c_end - c_start]
            except Exception as e:
                raise RuntimeError(f'Prediction failed at batch {batch_idx}') from e
    total_time = time.time() - start_time
    print(f'✅ 预测完成! 用时: {total_time / 60:.1f} 分钟')
    total_pixels = prediction.size
    class1_pixels = np.sum(prediction == 1)
    class1_ratio = class1_pixels / total_pixels
    avg_confidence = np.mean(confidence)
    print(f'📈 预测统计:')
    print(f'   ✓ 总像素数: {total_pixels:,}')
    print(f'   ✓ 类别1像素数: {class1_pixels:,} ({class1_ratio:.2%})')
    print(f'   ✓ 类别0像素数: {total_pixels - class1_pixels:,} ({1 - class1_ratio:.2%})')
    print(f'   ✓ 平均置信度: {avg_confidence:.3f}')
    print('🎨 生成彩色输出...')
    color_prediction = np.zeros((height, width, 3), dtype=np.uint8)
    color_prediction[prediction == 1] = [255, 0, 0]
    color_prediction[prediction == 0] = [0, 0, 0]
    print('💾 保存结果...')
    with rasterio.open(output_tif_path, 'w', driver='GTiff', height=height, width=width, count=3, dtype=np.uint8, crs=crs, transform=transform, compress='lzw') as dst:
        for i in range(3):
            dst.write(color_prediction[:, :, i], i + 1)
    confidence_path = output_tif_path.replace('.tif', '_confidence.tif')
    with rasterio.open(confidence_path, 'w', driver='GTiff', height=height, width=width, count=1, dtype=np.float32, crs=crs, transform=transform, compress='lzw') as dst:
        dst.write(confidence, 1)
    print(f'✅ 结果保存完成!')
    print(f'   ✓ 分割结果: {output_tif_path}')
    print(f'   ✓ 置信度图: {confidence_path}')
    predict_dataset.close()
    return (prediction, confidence)

def save_training_plots(train_losses, val_losses, val_accuracies, save_path):
    print('📈 绘制训练曲线...')
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='训练损失', color='blue', linewidth=2)
    plt.plot(val_losses, label='验证损失', color='red', linewidth=2)
    plt.title('损失曲线', fontsize=14, fontweight='bold')
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('损失值', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies, label='验证精度', color='green', linewidth=2)
    plt.title('精度曲线', fontsize=14, fontweight='bold')
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('精度 (%)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 3, 3)
    plt.plot(train_losses, label='训练损失', alpha=0.7)
    plt.plot(val_losses, label='验证损失', alpha=0.7)
    plt.plot([acc / 100 for acc in val_accuracies], label='验证精度(标准化)', alpha=0.7)
    plt.title('学习曲线综合', fontsize=14, fontweight='bold')
    plt.xlabel('轮次', fontsize=12)
    plt.ylabel('标准化值', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f'   ✅ 训练曲线已保存: {save_path}')

def print_final_report(train_losses, val_losses, val_accuracies, large_test_loss, large_test_accuracy, large_test_acc0, large_test_acc1, prediction, confidence, config):
    print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
    print('=' * 80)
    final_val_accuracy = max(val_accuracies) if val_accuracies else 0
    total_pixels = prediction.size
    class1_pixels = np.sum(prediction == 1)
    small_sample_ratio = config.get('small_sample_ratio', 0.2)
    val_ratio_in_small = config.get('val_ratio_in_small', 0.25)
    train_ratio = small_sample_ratio * (1 - val_ratio_in_small)
    val_ratio = small_sample_ratio * val_ratio_in_small
    large_test_ratio = 1.0 - small_sample_ratio
    print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
    print(f'   📊 数据使用策略:')
    print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
    print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
    print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
    print(f'   📈 性能表现:')
    print(f'       - 最佳验证精度: {final_val_accuracy:.2f}%')
    print(f'       - 大规模测试精度: {large_test_accuracy:.2f}% (Acc0={large_test_acc0:.1f}% | Acc1={large_test_acc1:.1f}%)')
    print(f'       - 训练轮数: {len(train_losses)}')
    print(f'       - 最终训练损失: {train_losses[-1]:.4f}')
    print(f'       - 最终验证损失: {val_losses[-1]:.4f}')
    print(f'       - 大测试集损失: {large_test_loss:.4f}')
    print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
    val_test_diff = abs(final_val_accuracy - large_test_accuracy)
    print(f'   📊 泛化性能评估:')
    if val_test_diff < 2.0:
        print(f'       ✅ 泛化能力优秀 (验证-大测试差异: {val_test_diff:.2f}%)')
    elif val_test_diff < 5.0:
        print(f'       ✅ 泛化能力良好 (验证-大测试差异: {val_test_diff:.2f}%)')
    elif val_test_diff < 10.0:
        print(f'       ⚠️ 存在轻微过拟合 (验证-大测试差异: {val_test_diff:.2f}%)')
    else:
        print(f'       ⚠️ 过拟合较明显 (验证-大测试差异: {val_test_diff:.2f}%)')
    print(f'   📈 样本效率评估:')
    if large_test_accuracy > 85:
        print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
    elif large_test_accuracy > 75:
        print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
    elif large_test_accuracy > 65:
        print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
    else:
        print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
    print(f'\n🔮 大规模预测结果分析:')
    print(f'   📊 预测覆盖范围: 整个研究区域')
    print(f'   📊 总像素数: {total_pixels:,}')
    print(f'   📊 检测到的目标像素: {class1_pixels:,} ({class1_pixels / total_pixels:.2%})')
    print(f'   📊 背景像素: {total_pixels - class1_pixels:,} ({1 - class1_pixels / total_pixels:.2%})')
    if confidence is not None:
        avg_conf = np.mean(confidence)
        high_conf_pixels = np.sum(confidence > 0.8)
        print(f'   📊 平均预测置信度: {avg_conf:.3f}')
        print(f'   📊 高置信度像素 (>0.8): {high_conf_pixels:,} ({high_conf_pixels / total_pixels:.2%})')
    print(f'\n🎯 实验意义与建议:')
    if large_test_accuracy > 80 and val_test_diff < 5:
        print(f'   ✅ 模型具有良好的泛化能力，可用于大规模预测')
        print('空间划分：可可西里内正负图块各自8:2训练/验证，其他地区仅测试；实际数量见spatial_split_audit.json')
    elif large_test_accuracy > 70:
        print(f'   📊 模型性能尚可，但可考虑增加训练样本提升精度')
        print(f'   📊 当前训练策略在数据受限场景下有一定价值')
    else:
        print(f'   ⚠️ 建议增加训练数据或调整模型架构')
        print(f'   ⚠️ 可尝试数据增强、迁移学习等技术')
    print(f'\n📁 输出文件:')
    print(f'   ✅ 模型文件: {config['model_save_path']}')
    print(f'   ✅ 大规模预测结果: {config['output_path']}')
    confidence_path = config['output_path'].replace('.tif', '_confidence.tif')
    print(f'   ✅ 置信度图: {confidence_path}')
    if config.get('save_plots'):
        plot_path = config.get('plot_save_path', 'training_curves.png')
        print(f'   ✅ 训练曲线: {plot_path}')

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

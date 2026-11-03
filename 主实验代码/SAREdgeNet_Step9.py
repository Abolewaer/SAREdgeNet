import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / '.runtime'), str(_ROOT)]
from saredge.legacy_io import RasterCV
import cv2
import numpy as np
import os
from scipy import ndimage
from skimage import measure
import time
from tqdm import tqdm

def vectorized_gap_analysis(labeled_gaps, gray_image, mean_threshold, std_threshold, small_gap_threshold):
    unique_labels = np.unique(labeled_gaps)[1:]
    if len(unique_labels) == 0:
        return (np.array([]), np.array([]), np.array([]))
    gap_sizes = np.zeros(len(unique_labels), dtype=np.int32)
    gap_means = np.zeros(len(unique_labels), dtype=np.float32)
    gap_stds = np.zeros(len(unique_labels), dtype=np.float32)
    for i, label in enumerate(tqdm(unique_labels, desc='分析空隙', leave=False)):
        mask = labeled_gaps == label
        pixels = gray_image[mask]
        gap_sizes[i] = np.sum(mask)
        gap_means[i] = np.mean(pixels)
        gap_stds[i] = np.std(pixels)
    small_gaps = gap_sizes <= small_gap_threshold
    condition_gaps = (gap_means <= mean_threshold) & (gap_stds <= std_threshold) & ~small_gaps
    fill_gaps = small_gaps | condition_gaps
    return (unique_labels[fill_gaps], gap_sizes[fill_gaps], small_gaps[fill_gaps])

def batch_process_contours(image, red_mask, contours, mean_threshold, std_threshold, small_gap_threshold, fill_color, batch_size=50):
    result_image = image.copy()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    total_filled = 0
    total_gaps_analyzed = 0
    small_gaps_filled = 0
    condition_gaps_filled = 0
    print(f'开始批量处理 {len(contours)} 个轮廓，批次大小: {batch_size}')
    num_batches = (len(contours) + batch_size - 1) // batch_size
    for batch_start in tqdm(range(0, len(contours), batch_size), desc='处理轮廓批次', total=num_batches):
        batch_end = min(batch_start + batch_size, len(contours))
        batch_contours = contours[batch_start:batch_end]
        for contour in tqdm(batch_contours, desc=f'批次{batch_start // batch_size + 1}', leave=False):
            contour_mask = np.zeros(red_mask.shape, dtype=np.uint8)
            cv2.fillPoly(contour_mask, [contour], 255)
            internal_gaps = (contour_mask > 0) & (red_mask == 0)
            if np.any(internal_gaps):
                labeled_gaps, num_gaps = ndimage.label(internal_gaps)
                if num_gaps > 0:
                    fill_labels, gap_sizes, is_small_gaps = vectorized_gap_analysis(labeled_gaps, gray_image, mean_threshold, std_threshold, small_gap_threshold)
                    total_gaps_analyzed += num_gaps
                    for i, label in enumerate(tqdm(fill_labels, desc='填充空隙', leave=False)):
                        mask = labeled_gaps == label
                        result_image[mask] = fill_color
                        total_filled += gap_sizes[i]
                        if is_small_gaps[i]:
                            small_gaps_filled += 1
                        else:
                            condition_gaps_filled += 1
    return (result_image, total_filled, total_gaps_analyzed, small_gaps_filled, condition_gaps_filled)

def detect_and_fill_internal_gaps_optimized(final_tif_path, output_path, mean_threshold=50, std_threshold=30, small_gap_threshold=20, use_green_fill=True, batch_size=50):
    start_time = time.time()
    print('=' * 60)
    print('智能空隙填充处理 (修正版 - 只填充轮廓内部)')
    print('=' * 60)
    print(f'均值阈值: {mean_threshold}')
    print(f'标准差阈值: {std_threshold}')
    print(f'小空隙阈值: {small_gap_threshold} 像素')
    print(f'填充颜色: {('绿色(调试模式)' if use_green_fill else '红色')}')
    print(f'批处理大小: {batch_size}')
    print('正在读取Final.tif...')
    image = cv2.imread(final_tif_path)
    if image is None:
        raise ValueError(f'无法读取Final.tif: {final_tif_path}')
    print(f'图像尺寸: {image.shape}')
    print('正在提取红色像素...')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    red_pixel_count = np.sum(red_mask > 0)
    print(f'检测到红色像素: {red_pixel_count} 个')
    if red_pixel_count == 0:
        print('警告: 未检测到红色像素，保存原图像')
        cv2.imwrite(output_path, image)
        return (image, 0, 0)
    print('正在分析红色轮廓...')
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    print(f'找到 {contour_count} 个红色轮廓')
    if contour_count == 0:
        print('警告: 无有效轮廓，保存原图像')
        cv2.imwrite(output_path, image)
        return (image, 0, 0)
    fill_color = [0, 255, 0] if use_green_fill else [0, 0, 255]
    color_name = '绿色' if use_green_fill else '红色'
    print('开始批量处理轮廓...')
    result_image, total_filled_pixels, total_gaps_analyzed, small_gaps_filled, condition_gaps_filled = batch_process_contours(image, red_mask, contours, mean_threshold, std_threshold, small_gap_threshold, fill_color, batch_size)
    filled_gaps_count = small_gaps_filled + condition_gaps_filled
    print('正在保存结果...')
    with tqdm(total=1, desc='保存文件') as pbar:
        save_params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
        success = cv2.imwrite(output_path, result_image, save_params)
        pbar.update(1)
    if success:
        print(f'Polish.tif已保存到: {output_path}')
        with tqdm(total=1, desc='验证文件') as pbar:
            saved_image = cv2.imread(output_path)
            pbar.update(1)
        if saved_image is not None:
            print(f'文件保存验证成功，尺寸: {saved_image.shape}')
        else:
            print('警告: 保存的文件无法读取')
    else:
        raise ValueError(f'保存失败: {output_path}')
    end_time = time.time()
    processing_time = end_time - start_time
    total_pixels = result_image.shape[0] * result_image.shape[1]
    fill_percentage = total_filled_pixels / total_pixels * 100 if total_pixels > 0 else 0
    print('\n' + '=' * 60)
    print('智能填充完成统计信息 (修正版)')
    print('=' * 60)
    print(f'处理时间: {processing_time:.2f} 秒')
    print(f'处理的轮廓数量: {contour_count}')
    print(f'分析的空隙总数: {total_gaps_analyzed}')
    print(f'填充的空隙数量: {filled_gaps_count}')
    print(f'  - 小空隙填充: {small_gaps_filled} 个')
    print(f'  - 条件填充: {condition_gaps_filled} 个')
    print(f'填充的像素总数: {total_filled_pixels:,}')
    print(f'填充像素占比: {fill_percentage:.4f}%')
    print(f'填充颜色: {color_name}')
    print(f'批处理大小: {batch_size}')
    print(f'输出文件大小: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB')
    return (result_image, total_filled_pixels, total_gaps_analyzed)
cv2 = RasterCV()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    detect_and_fill_internal_gaps_optimized(args.input, args.output, 50, 30, 20, False)
if __name__ == '__main__':
    main()

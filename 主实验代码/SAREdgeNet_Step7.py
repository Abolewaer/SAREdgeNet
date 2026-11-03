import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / '.runtime'), str(_ROOT)]
from saredge.legacy_io import RasterCV
import cv2
import numpy as np
import os
import time
import gc

def process_image_chunks(image, chunk_size=1024, overlap_ratio=0.3):
    height, width = image.shape[:2]
    chunks = []
    overlap = int(chunk_size * overlap_ratio)
    step_size = chunk_size - overlap
    print(f'块大小: {chunk_size}x{chunk_size}, 重叠: {overlap}像素 ({overlap_ratio * 100}%), 步长: {step_size}')
    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            y_end = min(y + chunk_size, height)
            x_end = min(x + chunk_size, width)
            chunks.append({'x': x, 'y': y, 'x_end': x_end, 'y_end': y_end, 'width': x_end - x, 'height': y_end - y, 'overlap': overlap})
    return chunks

def merge_overlapping_results(result_image, chunk_result, chunk_info, blend_overlap=True):
    y, x = (chunk_info['y'], chunk_info['x'])
    y_end, x_end = (chunk_info['y_end'], chunk_info['x_end'])
    overlap = chunk_info['overlap']
    if blend_overlap and overlap > 0:
        chunk_height, chunk_width = chunk_result.shape[:2]
        weight_mask = np.ones((chunk_height, chunk_width), dtype=np.float32)
        if x > 0:
            fade_width = min(overlap, chunk_width)
            for i in range(fade_width):
                weight_mask[:, i] = i / fade_width
        if y > 0:
            fade_height = min(overlap, chunk_height)
            for i in range(fade_height):
                weight_mask[i, :] = np.minimum(weight_mask[i, :], i / fade_height)
        if x_end < result_image.shape[1]:
            fade_width = min(overlap, chunk_width)
            for i in range(fade_width):
                col_idx = chunk_width - 1 - i
                if col_idx >= 0:
                    weight_mask[:, col_idx] = np.minimum(weight_mask[:, col_idx], i / fade_width)
        if y_end < result_image.shape[0]:
            fade_height = min(overlap, chunk_height)
            for i in range(fade_height):
                row_idx = chunk_height - 1 - i
                if row_idx >= 0:
                    weight_mask[row_idx, :] = np.minimum(weight_mask[row_idx, :], i / fade_height)
        existing_region = result_image[y:y_end, x:x_end].astype(np.float32)
        new_region = chunk_result.astype(np.float32)
        weight_mask_3d = np.stack([weight_mask, weight_mask, weight_mask], axis=2)
        blended = existing_region * (1 - weight_mask_3d) + new_region * weight_mask_3d
        result_image[y:y_end, x:x_end] = blended.astype(np.uint8)
    else:
        non_zero_mask = np.any(chunk_result > 0, axis=2)
        result_image[y:y_end, x:x_end][non_zero_mask] = chunk_result[non_zero_mask]

def detect_red_pixels_optimized(image_chunk):
    hsv = cv2.cvtColor(image_chunk, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    if np.sum(red_mask) == 0:
        lower_red_bgr = np.array([0, 0, 100])
        upper_red_bgr = np.array([100, 100, 255])
        red_mask = cv2.inRange(image_chunk, lower_red_bgr, upper_red_bgr)
    return red_mask

def remove_small_white_regions_opencv(image, min_area_threshold, chunk_size=1024):
    print(f'使用OpenCV优化模式过滤小区域，分块大小: {chunk_size}x{chunk_size}')
    height, width = image.shape[:2]
    filtered_image = np.zeros_like(image)
    chunks = process_image_chunks(image, chunk_size, overlap_ratio=0.3)
    print(f'将图像分为 {len(chunks)} 个块进行过滤处理')
    total_removed_regions = 0
    total_removed_pixels = 0
    for i, chunk_info in enumerate(chunks):
        print(f'过滤处理块 {i + 1}/{len(chunks)}')
        y, x = (chunk_info['y'], chunk_info['x'])
        y_end, x_end = (chunk_info['y_end'], chunk_info['x_end'])
        chunk = image[y:y_end, x:x_end]
        white_threshold = 200
        white_mask = cv2.inRange(chunk, (white_threshold, white_threshold, white_threshold), (255, 255, 255)).astype(np.uint8)
        if np.sum(white_mask) == 0:
            merge_overlapping_results(filtered_image, chunk, chunk_info, blend_overlap=False)
            continue
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
        filtered_chunk = chunk.copy()
        removed_regions = 0
        removed_pixels = 0
        for j in range(1, num_labels):
            area = stats[j, cv2.CC_STAT_AREA]
            if area < min_area_threshold:
                component_mask = labels == j
                filtered_chunk[component_mask] = [0, 0, 0]
                removed_regions += 1
                removed_pixels += area
        merge_overlapping_results(filtered_image, filtered_chunk, chunk_info, blend_overlap=False)
        total_removed_regions += removed_regions
        total_removed_pixels += removed_pixels
        del chunk, white_mask, labels, stats, filtered_chunk
        gc.collect()
    print(f'总共删除了 {total_removed_regions} 个小区域，共 {total_removed_pixels} 个像素')
    return filtered_image

def extract_red_contours_and_process_opencv_optimized(area_threshold=50, chunk_size=1024, distance_threshold=17, output_dir=None, source_path=None, target_path=None):
    print(f'使用OpenCV距离变换优化模式，分块大小: {chunk_size}x{chunk_size}')
    print(f'距离阈值: {distance_threshold} 像素')
    try:
        start_time = time.time()
        print('读取图像...')
        source_img = cv2.imread(source_path, cv2.IMREAD_COLOR)
        target_img = cv2.imread(target_path, cv2.IMREAD_COLOR)
        if source_img is None or target_img is None:
            raise ValueError('无法读取图像文件')
        print(f'源图像尺寸: {source_img.shape}')
        print(f'目标图像尺寸: {target_img.shape}')
        if source_img.shape != target_img.shape:
            source_img = cv2.resize(source_img, (target_img.shape[1], target_img.shape[0]))
        height, width = source_img.shape[:2]
        final_result = np.zeros_like(target_img)
        chunks = process_image_chunks(source_img, chunk_size, overlap_ratio=0.3)
        print(f'将图像分为 {len(chunks)} 个块进行处理')
        total_red_pixels = 0
        total_white_pixels = 0
        for i, chunk_info in enumerate(chunks):
            print(f'处理块 {i + 1}/{len(chunks)}')
            y, x = (chunk_info['y'], chunk_info['x'])
            y_end, x_end = (chunk_info['y_end'], chunk_info['x_end'])
            source_chunk = source_img[y:y_end, x:x_end]
            target_chunk = target_img[y:y_end, x:x_end]
            red_mask = detect_red_pixels_optimized(source_chunk)
            white_threshold = 200
            white_mask = cv2.inRange(target_chunk, (white_threshold, white_threshold, white_threshold), (255, 255, 255)) > 0
            blue_pixels = red_mask > 0
            if np.sum(blue_pixels) == 0:
                continue
            blue_binary = blue_pixels.astype(np.uint8) * 255
            blue_inverted = 255 - blue_binary
            dist_transform = cv2.distanceTransform(blue_inverted, cv2.DIST_L2, 5)
            dilated_region = dist_transform <= distance_threshold
            white_near_skeleton = white_mask & dilated_region
            result_chunk = np.zeros_like(target_chunk)
            result_chunk[blue_pixels] = [255, 0, 0]
            result_chunk[white_near_skeleton] = [255, 255, 255]
            merge_overlapping_results(final_result, result_chunk, chunk_info, blend_overlap=False)
            total_red_pixels += np.sum(blue_pixels)
            total_white_pixels += np.sum(white_near_skeleton)
            del source_chunk, target_chunk, red_mask, white_mask
            del blue_pixels, blue_binary, blue_inverted, dist_transform
            del dilated_region, white_near_skeleton, result_chunk
            gc.collect()
        print(f'检测到的红色像素总数: {total_red_pixels}')
        print(f'保留的白色像素总数: {total_white_pixels}')
        print(f'开始过滤小于 {area_threshold} 像素的白色区域...')
        final_result_filtered = remove_small_white_regions_opencv(final_result, area_threshold, chunk_size)
        processed_result_path = os.path.join(output_dir, 'processed_result_opencv_optimized.tif')
        cv2.imwrite(processed_result_path, final_result_filtered)
        print(f'处理完成！耗时: {time.time() - start_time:.2f} 秒')
        print(f'结果已保存到: {processed_result_path}')
        return (final_result_filtered, processed_result_path, total_red_pixels, total_white_pixels)
    except Exception as e:
        print(f'处理过程中出现错误: {str(e)}')
        return (None, None, 0, 0)

def extract_white_contours_and_overlay_optimized(processed_result_path, output_dir, target_overlay_path):
    try:
        start_time = time.time()
        print('读取图像进行覆盖处理...')
        processed_img = cv2.imread(processed_result_path, cv2.IMREAD_COLOR)
        target_img = cv2.imread(target_overlay_path, cv2.IMREAD_COLOR)
        if processed_img is None or target_img is None:
            raise ValueError('无法读取图像文件')
        print(f'处理结果图像尺寸: {processed_img.shape}')
        print(f'目标图像尺寸: {target_img.shape}')
        if processed_img.shape != target_img.shape:
            processed_img = cv2.resize(processed_img, (target_img.shape[1], target_img.shape[0]))
        white_threshold = 200
        white_mask = cv2.inRange(processed_img, (white_threshold, white_threshold, white_threshold), (255, 255, 255)) > 0
        result_img = target_img.copy()
        print('将白色轮廓用红色覆盖...')
        result_img[white_mask] = [0, 0, 255]
        final_output_path = os.path.join(output_dir, 'final_red_overlay_result_optimized.tif')
        cv2.imwrite(final_output_path, result_img)
        white_pixel_count = np.sum(white_mask)
        total_pixels = processed_img.shape[0] * processed_img.shape[1]
        coverage_percentage = white_pixel_count / total_pixels * 100
        print(f'白色轮廓像素数量: {white_pixel_count}')
        print(f'覆盖率: {coverage_percentage:.2f}%')
        print(f'覆盖处理完成！耗时: {time.time() - start_time:.2f} 秒')
        print(f'最终结果已保存到: {final_output_path}')
        return (result_img, white_pixel_count, coverage_percentage)
    except Exception as e:
        print(f'覆盖处理过程中出现错误: {str(e)}')
        return (None, 0, 0)

def create_process_summary_optimized(output_dir, red_pixels, white_pixels, area_threshold, distance_threshold, overlay_pixels, coverage):
    summary_path = os.path.join(output_dir, 'opencv_optimized_process_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('OpenCV优化版图像处理过程摘要\n')
        f.write('=' * 50 + '\n\n')
        f.write(f'处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n')
        f.write(f'滑窗设置: 30%重叠\n')
        f.write(f'优化特性: 使用OpenCV距离变换\n\n')
        f.write('第一步：红色轮廓提取和处理（OpenCV优化）\n')
        f.write(f'- 检测到的红色像素总数: {red_pixels}\n')
        f.write(f'- 保留的白色像素总数: {white_pixels}\n')
        f.write(f'- 面积阈值: {area_threshold} 像素\n')
        f.write(f'- 距离阈值: {distance_threshold} 像素\n')
        f.write(f'- 使用OpenCV距离变换代替传统膨胀操作\n\n')
        f.write('第二步：白色轮廓红色覆盖\n')
        f.write(f'- 覆盖的像素数量: {overlay_pixels}\n')
        f.write(f'- 覆盖率: {coverage:.2f}%\n\n')
        f.write('性能优化:\n')
        f.write('- 使用cv2.distanceTransform()替代手动距离计算\n')
        f.write('- 使用cv2.inRange()替代逐像素比较\n')
        f.write('- 优化的内存管理和垃圾回收\n')
        f.write('- 分块处理避免大图像内存溢出\n\n')
        f.write('输出文件:\n')
        f.write('- processed_result_opencv_optimized.tif: 中间处理结果\n')
        f.write('- final_red_overlay_result_optimized.tif: 最终红色覆盖结果\n')
        f.write('- opencv_optimized_process_summary.txt: 处理摘要\n')
    print(f'OpenCV优化处理摘要已保存到: {summary_path}')
cv2 = RasterCV()

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--contours', required=True)
    parser.add_argument('--support', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--seed-max-intensity',type=float,default=50.0)
    parser.add_argument('--seed-tolerance',type=float,default=30.0)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--chunk-size', type=int, default=4096)
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    result, path, _, _ = extract_red_contours_and_process_opencv_optimized(1200, args.chunk_size, 17, args.output_dir, args.contours, args.support)
    if result is None:
        raise RuntimeError('Boundary support processing failed')
    result, _, _ = extract_white_contours_and_overlay_optimized(path, args.output_dir, args.contours)
    if result is None:
        raise RuntimeError('Support overlay failed')
    from saredge.region_grow import seed_grow
    seed_grow(Path(args.output_dir)/'final_red_overlay_result_optimized.tif',args.source,
              Path(args.output_dir)/'SeedGrow.tif',args.seed_max_intensity,args.seed_tolerance)
if __name__ == '__main__':
    main()

import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT / '.runtime'), str(_ROOT)]
from saredge.legacy_io import RasterCV
import cv2
import numpy as np
from PIL import Image
import os
import time

def extract_red_pixels(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f'无法读取图像: {image_path}')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    red_pixels = cv2.bitwise_and(image, image, mask=red_mask)
    return (red_pixels, red_mask)

def subtract_red_pixels(seed_grow_path, final_red_path, output_path):
    _, seed_mask = extract_red_pixels(seed_grow_path)
    _, final_mask = extract_red_pixels(final_red_path)
    result_mask = cv2.subtract(seed_mask, final_mask)
    original = cv2.imread(seed_grow_path)
    result_image = np.zeros_like(original)
    result_image[result_mask > 0] = [0, 0, 255]
    cv2.imwrite(output_path, result_image)
    return (result_image, result_mask)

def apply_sobel_edge_detection(image_path, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f'无法读取图像: {image_path}')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_magnitude = np.uint8(sobel_magnitude / sobel_magnitude.max() * 255 if sobel_magnitude.max() > 0 else sobel_magnitude)
    blue_mask = np.zeros_like(image)
    blue_mask[sobel_magnitude > 50] = [255, 0, 0]
    return (blue_mask, sobel_magnitude)

def overlay_and_extract_colors(base_image_path, blue_mask, output_path):
    base_image = cv2.imread(base_image_path)
    if base_image is None:
        raise ValueError(f'无法读取图像: {base_image_path}')
    overlay_image = cv2.addWeighted(base_image, 0.7, blue_mask, 0.3, 0)
    hsv = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask_hsv = cv2.inRange(hsv, lower_blue, upper_blue)
    combined_mask = cv2.bitwise_or(red_mask, blue_mask_hsv)
    result_image = np.zeros_like(overlay_image)
    result_image[red_mask > 0] = [0, 0, 255]
    result_image[blue_mask_hsv > 0] = [255, 0, 0]
    cv2.imwrite(output_path, result_image)
    return result_image

def filter_red_pixels_by_distance_optimized(image_path, output_path, distance_threshold=50):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f'无法读取图像: {image_path}')
    print(f'开始处理图像: {image_path}')
    print(f'蓝色锚点距离阈值: {distance_threshold}')
    print(f'图像尺寸: {image.shape}')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    blue_pixel_count = np.sum(blue_mask > 0)
    red_pixel_count = np.sum(red_mask > 0)
    print(f'找到蓝色像素: {blue_pixel_count} 个')
    print(f'找到红色像素: {red_pixel_count} 个')
    result_image = np.zeros_like(image)
    result_image[blue_mask > 0] = [255, 0, 0]
    if blue_pixel_count == 0:
        print('警告: 没有找到蓝色像素，无法进行距离过滤')
        result_image[red_mask > 0] = [0, 0, 255]
        cv2.imwrite(output_path, result_image)
        return (result_image, red_mask)
    if red_pixel_count == 0:
        print('警告: 没有找到红色像素')
        cv2.imwrite(output_path, result_image)
        return (result_image, np.zeros_like(red_mask))
    print('开始距离变换计算...')
    blue_mask_inverted = 255 - blue_mask
    dist_transform = cv2.distanceTransform(blue_mask_inverted, cv2.DIST_L2, 5)
    print('距离变换计算完成')
    filtered_red_mask = (red_mask > 0) & (dist_transform <= distance_threshold)
    filtered_red_count = np.sum(filtered_red_mask)
    print(f'过滤后保留的红色像素: {filtered_red_count} 个')
    print(f'过滤掉的红色像素: {red_pixel_count - filtered_red_count} 个')
    result_image[filtered_red_mask] = [0, 0, 255]
    cv2.imwrite(output_path, result_image)
    print(f'过滤后的图像已保存到: {output_path}')
    return (result_image, filtered_red_mask)

def extract_red_contours_only(filtered_image_path, output_path):
    image = cv2.imread(filtered_image_path)
    if image is None:
        raise ValueError(f'无法读取图像: {filtered_image_path}')
    print(f'从 {filtered_image_path} 提取红色轮廓...')
    print(f'图像尺寸: {image.shape}')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    red_pixel_count = np.sum(red_mask > 0)
    print(f'找到红色像素: {red_pixel_count} 个')
    result_image = np.zeros_like(image)
    result_image[red_mask > 0] = [0, 0, 255]
    cv2.imwrite(output_path, result_image)
    print(f'红色轮廓图像已保存到: {output_path}')
    return (result_image, red_mask)

def overlay_red_as_green_and_filter_then_convert_to_red(red_tif_path, final_red_overlay_path, output_path, green_distance_threshold=30):
    red_image = cv2.imread(red_tif_path)
    if red_image is None:
        raise ValueError(f'无法读取Red.tif: {red_tif_path}')
    final_red_image = cv2.imread(final_red_overlay_path)
    if final_red_image is None:
        raise ValueError(f'无法读取final_red_overlay_result.tif: {final_red_overlay_path}')
    print(f'步骤6: 将Red.tif覆盖到final_red_overlay_result.tif上...')
    print(f'绿色锚点距离阈值: {green_distance_threshold}')
    if red_image.shape != final_red_image.shape:
        print(f'调整图像尺寸: Red.tif {red_image.shape} -> {final_red_image.shape}')
        red_image = cv2.resize(red_image, (final_red_image.shape[1], final_red_image.shape[0]))
    hsv_red = cv2.cvtColor(red_image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    red_mask_from_red_tif = cv2.bitwise_or(cv2.inRange(hsv_red, lower_red1, upper_red1), cv2.inRange(hsv_red, lower_red2, upper_red2))
    overlay_image = final_red_image.copy()
    overlay_image[red_mask_from_red_tif > 0] = [0, 255, 0]
    print('覆盖完成，开始提取红色和绿色轮廓...')
    hsv_overlay = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    green_mask = cv2.inRange(hsv_overlay, lower_green, upper_green)
    red_mask_overlay = cv2.bitwise_or(cv2.inRange(hsv_overlay, lower_red1, upper_red1), cv2.inRange(hsv_overlay, lower_red2, upper_red2))
    green_pixel_count = np.sum(green_mask > 0)
    red_pixel_count = np.sum(red_mask_overlay > 0)
    print(f'找到绿色像素（锚点）: {green_pixel_count} 个')
    print(f'找到红色像素: {red_pixel_count} 个')
    result_image = np.zeros_like(overlay_image)
    result_image[green_mask > 0] = [0, 255, 0]
    if green_pixel_count == 0:
        print('警告: 没有找到绿色像素，无法进行距离过滤')
        result_image[red_mask_overlay > 0] = [0, 0, 255]
    elif red_pixel_count > 0:
        print('开始对红色像素进行距离变换过滤...')
        green_mask_inverted = 255 - green_mask
        dist_transform = cv2.distanceTransform(green_mask_inverted, cv2.DIST_L2, 5)
        filtered_red_mask = (red_mask_overlay > 0) & (dist_transform <= green_distance_threshold)
        filtered_red_count = np.sum(filtered_red_mask)
        print(f'过滤后保留的红色像素: {filtered_red_count} 个')
        print(f'过滤掉的红色像素: {red_pixel_count - filtered_red_count} 个')
        result_image[filtered_red_mask] = [0, 0, 255]
    print('将绿色轮廓转换为红色...')
    result_image[green_mask > 0] = [0, 0, 255]
    cv2.imwrite(output_path, result_image)
    print(f'BurrDelect.tif已保存到: {output_path}（绿色轮廓已转换为红色，无缝隙）')
    return result_image

def overlay_on_processed_image(burr_detect_path, processed_image_path, output_path):
    burr_image = cv2.imread(burr_detect_path)
    if burr_image is None:
        raise ValueError(f'无法读取BurrDelect.tif: {burr_detect_path}')
    processed_image = cv2.imread(processed_image_path)
    if processed_image is None:
        raise ValueError(f'无法读取processed_image_adjusted.tif: {processed_image_path}')
    print(f'步骤7: 将BurrDelect.tif覆盖到processed_image_adjusted.tif上...')
    print(f'BurrDelect图像尺寸: {burr_image.shape}')
    print(f'processed_image图像尺寸: {processed_image.shape}')
    if burr_image.shape != processed_image.shape:
        print(f'调整BurrDelect图像尺寸: {burr_image.shape} -> {processed_image.shape}')
        burr_image = cv2.resize(burr_image, (processed_image.shape[1], processed_image.shape[0]))
    burr_gray = cv2.cvtColor(burr_image, cv2.COLOR_BGR2GRAY)
    burr_mask = burr_gray > 10
    result_image = processed_image.copy()
    result_image[burr_mask] = burr_image[burr_mask]
    overlay_pixel_count = np.sum(burr_mask)
    print(f'覆盖的像素数量: {overlay_pixel_count} 个')
    cv2.imwrite(output_path, result_image)
    print(f'Step7.tif已保存到: {output_path}')
    return result_image

def overlay_red_pixels_from_dilite_to_step7(dilite_path, step7_path, output_path):
    start_time = time.time()
    print('=' * 60)
    print('步骤8: 将dilite.tif中的红色像素覆盖到Step7.tif上')
    print('=' * 60)
    print('正在读取图像...')
    dilite_image = cv2.imread(dilite_path)
    step7_image = cv2.imread(step7_path)
    if dilite_image is None:
        raise ValueError(f'无法读取dilite.tif: {dilite_path}')
    if step7_image is None:
        raise ValueError(f'无法读取Step7.tif: {step7_path}')
    print(f'dilite.tif 尺寸: {dilite_image.shape}')
    print(f'Step7.tif 尺寸: {step7_image.shape}')
    if dilite_image.shape != step7_image.shape:
        print(f'调整dilite.tif尺寸: {dilite_image.shape} -> {step7_image.shape}')
        dilite_image = cv2.resize(dilite_image, (step7_image.shape[1], step7_image.shape[0]))
        print('尺寸调整完成')
    print('正在检测红色像素...')
    hsv = cv2.cvtColor(dilite_image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    red_pixel_count = np.sum(red_mask > 0)
    print(f'检测到红色像素: {red_pixel_count} 个')
    print('正在进行像素覆盖...')
    result_image = step7_image.copy()
    if red_pixel_count > 0:
        result_image[red_mask > 0] = dilite_image[red_mask > 0]
        print(f'成功覆盖 {red_pixel_count} 个红色像素')
    else:
        print('警告: 未检测到红色像素，将保存未修改的Step7.tif')
    print('正在保存结果...')
    success = cv2.imwrite(output_path, result_image)
    if success:
        print(f'Final.tif已保存到: {output_path}')
    else:
        raise ValueError(f'保存失败: {output_path}')
    end_time = time.time()
    processing_time = end_time - start_time
    total_pixels = result_image.shape[0] * result_image.shape[1]
    coverage_percentage = red_pixel_count / total_pixels * 100 if total_pixels > 0 else 0
    print('\n' + '=' * 60)
    print('步骤8完成统计信息')
    print('=' * 60)
    print(f'处理时间: {processing_time:.2f} 秒')
    print(f'图像总像素数: {total_pixels:,}')
    print(f'覆盖的红色像素数: {red_pixel_count:,}')
    print(f'红色像素覆盖率: {coverage_percentage:.4f}%')
    print(f'输出文件大小: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB')
    return (result_image, red_pixel_count, coverage_percentage)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed-grow')
    parser.add_argument('--step7-dir', required=True)
    parser.add_argument('--contours', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    '\n    主函数，执行完整的图像处理流程\n    '
    seed_grow_path = args.seed_grow or str(Path(args.step7_dir)/'SeedGrow.tif')
    final_red_path = str(Path(args.step7_dir) / 'final_red_overlay_result_optimized.tif')
    dilite_path = str(Path(args.output_dir) / 'dilite.tif')
    blue_contours_path = args.contours
    processed_image_path = args.source
    output_dir = args.output_dir
    blue_red_path = str(Path(args.output_dir) / 'BlueRed.tif')
    filtered_output_path = str(Path(args.output_dir) / 'BlueRed_filtered.tif')
    red_only_path = str(Path(args.output_dir) / 'Red.tif')
    burr_detect_path = str(Path(args.output_dir) / 'BurrDelect.tif')
    step7_path = str(Path(args.output_dir) / 'Step7.tif')
    final_path = str(Path(args.output_dir) / 'Final.tif')
    os.makedirs(output_dir, exist_ok=True)
    blue_distance_threshold = 20
    green_distance_threshold = 10
    try:
        print('步骤1: 提取红色像素并计算差异...')
        dilite_image, dilite_mask = subtract_red_pixels(seed_grow_path, final_red_path, dilite_path)
        print(f'dilite.tif 已保存到: {dilite_path}')
        print('步骤2: 进行Sobel边缘检测...')
        blue_mask, sobel_edges = apply_sobel_edge_detection(dilite_path, output_dir)
        print('Sobel边缘检测完成')
        print('步骤3: 叠加蓝色掩膜并提取红色和蓝色像素...')
        final_result = overlay_and_extract_colors(blue_contours_path, blue_mask, blue_red_path)
        print(f'BlueRed.tif 已保存到: {blue_red_path}')
        print('步骤4: 使用距离变换进行红色像素过滤...')
        filtered_result, filtered_red_mask = filter_red_pixels_by_distance_optimized(blue_red_path, filtered_output_path, blue_distance_threshold)
        print(f'过滤后的图像已保存到: {filtered_output_path}')
        print('步骤5: 提取红色轮廓...')
        red_only_result, red_mask = extract_red_contours_only(filtered_output_path, red_only_path)
        print(f'红色轮廓图像已保存到: {red_only_path}')
        print('步骤6: 距离过滤后将绿色轮廓转换为红色...')
        burr_detect_result = overlay_red_as_green_and_filter_then_convert_to_red(red_only_path, final_red_path, burr_detect_path, green_distance_threshold)
        print(f'BurrDelect.tif已保存到: {burr_detect_path}')
        print('步骤7: 将BurrDelect.tif覆盖在processed_image_adjusted.tif上...')
        step7_result = overlay_on_processed_image(burr_detect_path, processed_image_path, step7_path)
        print(f'Step7.tif已保存到: {step7_path}')
        print('步骤8: 将dilite.tif中的红色像素覆盖到Step7.tif上...')
        final_result, final_red_pixels, final_coverage = overlay_red_pixels_from_dilite_to_step7(dilite_path, step7_path, final_path)
        print(f'Final.tif已保存到: {final_path}')
        print('所有处理步骤完成！')
        print(f'\n输出文件总览:')
        print(f'- dilite.tif: {dilite_path}')
        print(f'- BlueRed.tif: {blue_red_path}')
        print(f'- BlueRed_filtered.tif: {filtered_output_path}')
        print(f'- Red.tif: {red_only_path}')
        print(f'- BurrDelect.tif: {burr_detect_path}')
        print(f'- Step7.tif: {step7_path}')
        print(f'- Final.tif: {final_path} (最终结果)')
        print(f'\n距离阈值设置:')
        print(f'- 蓝色锚点距离阈值: {blue_distance_threshold}')
        print(f'- 绿色锚点距离阈值: {green_distance_threshold}')
        print('- 最终结果：所有步骤处理后的完整图像')
        summary_path = os.path.join(output_dir, 'complete_processing_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('完整图像处理流程摘要\n')
            f.write('=' * 50 + '\n\n')
            f.write(f'处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n')
            f.write(f'完成所有8个处理步骤\n\n')
            f.write('输出文件列表:\n')
            f.write(f'1. dilite.tif - 红色像素差异结果\n')
            f.write(f'2. BlueRed.tif - 蓝红像素提取结果\n')
            f.write(f'3. BlueRed_filtered.tif - 距离过滤结果\n')
            f.write(f'4. Red.tif - 纯红色轮廓\n')
            f.write(f'5. BurrDelect.tif - 毛刺检测结果\n')
            f.write(f'6. Step7.tif - 中间处理结果\n')
            f.write(f'7. Final.tif - 最终完整结果\n\n')
            f.write(f'最终覆盖统计:\n')
            f.write(f'- 覆盖的红色像素数: {final_red_pixels:,}\n')
            f.write(f'- 红色像素覆盖率: {final_coverage:.4f}%\n')
        print(f'\n处理摘要已保存到: {summary_path}')
    except:
        raise
cv2 = RasterCV()
if __name__ == '__main__':
    main()

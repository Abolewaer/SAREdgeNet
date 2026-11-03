import rasterio
import geopandas as gpd
import numpy as np
from rasterio.mask import mask
from rasterio.windows import Window
import rasterio.features
import os
from tqdm import tqdm
import time
import psutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import warnings

warnings.filterwarnings('ignore')

                  
try:
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    print("💡 提示: 安装numba可进一步提升性能: pip install numba")


def get_memory_usage():

    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024         


def format_time(seconds):

    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


                         
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, fastmath=True)
    def process_pixels_fast(data, mask, lower_thresh, upper_thresh, nodata_val):

        height, width = data.shape
        result = np.zeros((3, height, width), dtype=np.uint8)
        valid_count = 0

        for i in prange(height):
            for j in prange(width):
                pixel_val = data[i, j]
                                                 
                if (pixel_val != nodata_val and
                        lower_thresh <= pixel_val <= upper_thresh):
                    result[0, i, j] = 255       
                    result[1, i, j] = 0         
                    result[2, i, j] = 0        
                    valid_count += 1

        return result, valid_count
else:
    def process_pixels_fast(data, mask, lower_thresh, upper_thresh, nodata_val):

                                      
        if nodata_val is not None:
            valid_data = data != nodata_val
        else:
            valid_data = np.ones_like(data, dtype=bool)

                                 
        in_range = (data >= lower_thresh) & (data <= upper_thresh)
        final_mask = valid_data & in_range                  

                 
        height, width = data.shape
        result = np.zeros((3, height, width), dtype=np.uint8)

                   
        result[0][final_mask] = 255       
        result[1][final_mask] = 0         
        result[2][final_mask] = 0        

        valid_count = np.sum(final_mask)
        return result, valid_count


def process_pixels_debug(data, mask, lower_thresh, upper_thresh, nodata_val, block_idx=None):


              
    total_pixels = data.size
    mask_pixels = np.sum(mask == 1)

    if nodata_val is not None:
        valid_data_pixels = np.sum(data != nodata_val)
        nodata_pixels = np.sum(data == nodata_val)
        valid_data_mask = data != nodata_val
    else:
        valid_data_pixels = total_pixels
        nodata_pixels = 0
        valid_data_mask = np.ones_like(data, dtype=bool)

                                
    mask_valid_data = (mask == 1) & valid_data_mask
    mask_valid_count = np.sum(mask_valid_data)

                             
    in_range_mask = (data >= lower_thresh) & (data <= upper_thresh)
    total_in_range = np.sum(valid_data_mask & in_range_mask)

                              
    mask_in_range = np.sum(mask_valid_data & in_range_mask)

                
    result, valid_count = process_pixels_fast(data, mask, lower_thresh, upper_thresh, nodata_val)

            
    if block_idx is not None and (mask_pixels > 0 or valid_count > 0):
        print(f"\n🔍 块 {block_idx} 调试信息:")
        print(f"   📊 总像素数: {total_pixels}")
        print(f"   🎯 mask区域像素: {mask_pixels}")
        print(f"   ✅ 有效数据像素: {valid_data_pixels}")
        print(f"   ❌ nodata像素: {nodata_pixels}")
        print(f"   🎯✅ mask+有效数据: {mask_valid_count}")
        print(f"   📈 mask区域内阈值范围: {mask_in_range}")
        print(f"   🔴 整个块阈值范围内: {total_in_range}")
        print(f"   🔴 最终红色像素: {valid_count}")

        if mask_valid_count > 0:
            mask_pixels_data = data[mask_valid_data]
            data_min = np.min(mask_pixels_data)
            data_max = np.max(mask_pixels_data)
            data_mean = np.mean(mask_pixels_data)
            print(f"   📊 mask区域数据范围: {data_min:.4f} ~ {data_max:.4f} (均值: {data_mean:.4f})")

        print(f"   📊 阈值范围: {lower_thresh:.4f} ~ {upper_thresh:.4f}")

    return result, valid_count


def create_global_mask(tif_path, gdf, temp_mask_path=None, chunk_size=4096):



    print("🎯 创建全局mask (这是一次性操作)...")

    with rasterio.open(tif_path) as src:
        profile = src.profile.copy()
        width = src.width
        height = src.height
        transform = src.transform

                      
    mask_profile = profile.copy()
    mask_profile.update({
        'count': 1,
        'dtype': 'uint8',
        'nodata': 0,
        'compress': 'lzw'           
    })

                       
    if temp_mask_path is None:
        temp_mask_path = tempfile.mktemp(suffix='_mask.tif')

    print(f"📁 临时mask文件: {temp_mask_path}")

             
    shapes = [(geom, 1) for geom in gdf.geometry]

                   
    n_chunks_x = (width + chunk_size - 1) // chunk_size
    n_chunks_y = (height + chunk_size - 1) // chunk_size
    total_chunks = n_chunks_x * n_chunks_y

    print(f"📊 mask创建信息:")
    print(f"   🧩 分块大小: {chunk_size} × {chunk_size}")
    print(f"   📊 分块数量: {total_chunks:,} 个")

    with rasterio.open(temp_mask_path, 'w', **mask_profile) as mask_dst:
        with tqdm(total=total_chunks,
                  desc="🎯 创建mask",
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
                             '[⏱️{elapsed}<⏰{remaining}, 🚀{rate_fmt}]',
                  unit='块',
                  colour='cyan') as pbar:

            for i in range(n_chunks_y):
                for j in range(n_chunks_x):
                    col_start = j * chunk_size
                    row_start = i * chunk_size
                    col_end = min(col_start + chunk_size, width)
                    row_end = min(row_start + chunk_size, height)

                    window = Window(col_start, row_start,
                                    col_end - col_start, row_end - row_start)

                                     
                    chunk_transform = rasterio.windows.transform(window, transform)

                                
                    chunk_mask = rasterio.features.rasterize(
                        shapes,
                        out_shape=(row_end - row_start, col_end - col_start),
                        transform=chunk_transform,
                        fill=0,
                        dtype='uint8'
                    )

                            
                    mask_dst.write(chunk_mask, 1, window=window)
                    pbar.update(1)

    print(f"✅ 全局mask创建完成")
    return temp_mask_path


def calculate_stats_optimized(tif_path, mask_path, block_size=2048, debug=False):



    print("📊 计算统计值 (使用预计算mask)...")

    all_valid_pixels = []
    total_mask_pixels = 0
    total_data_pixels = 0
    blocks_with_data = 0

    with rasterio.open(tif_path) as src, rasterio.open(mask_path) as mask_src:
        width = src.width
        height = src.height
        nodata_val = src.nodata

        print(f"🔍 调试信息:")
        print(f"   📊 TIF nodata值: {nodata_val}")
        print(f"   📊 TIF数据类型: {src.dtypes[0]}")

        n_blocks_x = (width + block_size - 1) // block_size
        n_blocks_y = (height + block_size - 1) // block_size
        total_blocks = n_blocks_x * n_blocks_y

        with tqdm(total=total_blocks,
                  desc="📊 统计块",
                  bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
                             '[⏱️{elapsed}<⏰{remaining}, 🚀{rate_fmt}]',
                  unit='块',
                  colour='blue') as pbar:

            for i in range(n_blocks_y):
                for j in range(n_blocks_x):
                    col_start = j * block_size
                    row_start = i * block_size
                    col_end = min(col_start + block_size, width)
                    row_end = min(row_start + block_size, height)

                    window = Window(col_start, row_start,
                                    col_end - col_start, row_end - row_start)

                                 
                    block_data = src.read(1, window=window)
                    block_mask = mask_src.read(1, window=window)

                          
                    mask_pixels_in_block = np.sum(block_mask == 1)
                    total_mask_pixels += mask_pixels_in_block

                    if nodata_val is not None:
                        valid_data_mask = block_data != nodata_val
                    else:
                        valid_data_mask = np.ones_like(block_data, dtype=bool)

                    data_pixels_in_block = np.sum(valid_data_mask)
                    total_data_pixels += data_pixels_in_block

                                                  
                    combined_mask = (block_mask == 1) & valid_data_mask

                    if np.any(combined_mask):
                        blocks_with_data += 1
                        valid_pixels = block_data[combined_mask]
                        all_valid_pixels.extend(valid_pixels.flatten())

                                    
                        if debug and len(all_valid_pixels) <= len(valid_pixels):
                            print(f"\n🔍 第一个有效块 [{i},{j}]:")
                            print(f"   mask像素数: {mask_pixels_in_block}")
                            print(f"   数据像素数: {data_pixels_in_block}")
                            print(f"   有效像素数: {len(valid_pixels)}")
                            print(f"   数据范围: {np.min(valid_pixels):.4f} ~ {np.max(valid_pixels):.4f}")

                    pbar.set_postfix_str(f"累计{len(all_valid_pixels):,}像素 mask区域{total_mask_pixels:,}")
                    pbar.update(1)

            
    print(f"\n🔍 详细统计报告:")
    print(f"   📊 总mask像素数: {total_mask_pixels:,}")
    print(f"   📊 总有效数据像素数: {total_data_pixels:,}")
    print(f"   📊 有数据的块数: {blocks_with_data}/{total_blocks}")
    print(f"   📊 最终有效像素数: {len(all_valid_pixels):,}")

    if len(all_valid_pixels) == 0:
        print("❌ 问题诊断:")
        if total_mask_pixels == 0:
            print("   🔍 SHP区域mask为空 - 检查SHP文件和坐标系")
        elif total_data_pixels == 0:
            print("   🔍 TIF数据全为nodata - 检查TIF文件")
        else:
            print("   🔍 SHP区域和有效数据没有重叠 - 检查空间关系")
        raise ValueError("在SHP区域内没有找到有效像素")

    all_valid_pixels = np.array(all_valid_pixels)
    mean_val = np.mean(all_valid_pixels)
    std_val = np.std(all_valid_pixels)

                
    min_val = np.min(all_valid_pixels)
    max_val = np.max(all_valid_pixels)
    median_val = np.median(all_valid_pixels)

    print(f"✅ 统计计算完成:")
    print(f"   📊 最小值: {min_val:.4f}")
    print(f"   📊 最大值: {max_val:.4f}")
    print(f"   📊 均值: {mean_val:.4f}")
    print(f"   📊 中位数: {median_val:.4f}")
    print(f"   📊 标准差: {std_val:.4f}")
    print(f"   📊 均值-2σ: {mean_val - 2 * std_val:.4f}")
    print(f"   📊 均值+2σ: {mean_val + 2 * std_val:.4f}")
    print(f"   📊 有效像素数: {len(all_valid_pixels):,}")

                
    lower_thresh = mean_val - 2 * std_val
    upper_thresh = mean_val + 2 * std_val

    pixels_in_range = np.sum((all_valid_pixels >= lower_thresh) & (all_valid_pixels <= upper_thresh))
    percentage_in_range = pixels_in_range / len(all_valid_pixels) * 100

    print(f"   📈 均值±2σ范围内像素: {pixels_in_range:,} ({percentage_in_range:.1f}%)")

    if percentage_in_range < 90:
        print(f"⚠️  注意: 只有{percentage_in_range:.1f}%的像素在±2σ范围内，这可能不正常")
        print(f"   💡 建议检查数据分布或考虑调整阈值范围")

    return mean_val, std_val


def process_tif_with_shp_optimized(tif_path, shp_path, output_path, block_size=2048, debug_mode=True):



    start_time = time.time()
    print("=" * 60)
    print("🚀 开始高性能地理数据处理...")
    print("=" * 60)

                
    print("\n📁 步骤1: 读取SHP文件...")
    gdf = gpd.read_file(shp_path)
    print(f"✅ SHP文件加载完成，包含 {len(gdf)} 个要素")

    if debug_mode:
        print(f"🔍 SHP调试信息:")
        print(f"   📊 SHP坐标系: {gdf.crs}")
        print(f"   📊 SHP边界: {gdf.bounds}")
        print(f"   📊 几何体类型: {gdf.geom_type.unique()}")

                  
    print("\n📊 步骤2: 读取TIF文件信息...")
    with rasterio.open(tif_path) as src:
        profile = src.profile.copy()
        transform = src.transform
        bounds = src.bounds
        width = src.width
        height = src.height
        crs = src.crs
        nodata = src.nodata

    print(f"✅ TIF文件信息:")
    print(f"   📏 尺寸: {width:,} × {height:,} 像素")
    print(f"   📦 数据类型: {profile['dtype']}")
    print(f"   🔍 nodata值: {nodata}")

    if debug_mode:
        print(f"🔍 TIF调试信息:")
        print(f"   📊 TIF坐标系: {crs}")
        print(f"   📊 TIF边界: {bounds}")
        print(f"   📊 像素分辨率: {abs(transform.a):.6f} × {abs(transform.e):.6f}")

              
    print("\n🔄 步骤3: 检查坐标系...")
    if gdf.crs != crs:
        print(f"🔄 转换坐标系: {gdf.crs} → {crs}")
        gdf = gdf.to_crs(crs)
        if debug_mode:
            print(f"✅ 坐标系转换完成")
    else:
        print(f"✅ 坐标系匹配: {crs}")

                         
    print("\n🎯 步骤4: 创建全局mask...")
    temp_mask_path = create_global_mask(tif_path, gdf, chunk_size=4096)

              
    print("\n📈 步骤5: 计算统计值...")
    mean_val, std_val = calculate_stats_optimized(tif_path, temp_mask_path, block_size, debug=debug_mode)

    lower_threshold = mean_val - 2 * std_val
    upper_threshold = mean_val + 2 * std_val

    print(f"📊 处理阈值:")
    print(f"   📊 下限 (均值-2σ): {lower_threshold:.4f}")
    print(f"   📊 上限 (均值+2σ): {upper_threshold:.4f}")
    print(f"   📊 阈值范围宽度: {upper_threshold - lower_threshold:.4f}")

               
    print("\n⚙️ 步骤6: 配置输出...")
    output_profile = profile.copy()
    output_profile.update({
        'count': 3,
        'dtype': 'uint8',
        'nodata': 0,
        'compress': 'lzw'
    })

               
    print(f"\n🚀 步骤7: 高速分块处理 (块大小: {block_size}×{block_size})...")

    with rasterio.open(tif_path) as src, rasterio.open(temp_mask_path) as mask_src:
        with rasterio.open(output_path, 'w', **output_profile) as dst:

            n_blocks_x = (width + block_size - 1) // block_size
            n_blocks_y = (height + block_size - 1) // block_size
            total_blocks = n_blocks_x * n_blocks_y

            print(f"📋 处理信息:")
            print(f"   🧩 分块大小: {block_size} × {block_size} 像素")
            print(f"   📊 分块数量: {n_blocks_x} × {n_blocks_y} = {total_blocks:,} 个块")
            print(f"   💾 预估内存使用: ~{(block_size * block_size * 8 / 1024 / 1024):.1f}MB/块")

            processed_pixels = 0
            valid_pixels_found = 0
            start_processing_time = time.time()
            debug_blocks_shown = 0
            max_debug_blocks = 5                    

            with tqdm(total=total_blocks,
                      desc="🚀 高速处理",
                      bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} '
                                 '[⏱️{elapsed}<⏰{remaining}, 🚀{rate_fmt}]',
                      unit='块',
                      colour='green',
                      smoothing=0.1) as pbar:

                for i in range(n_blocks_y):
                    for j in range(n_blocks_x):
                        col_start = j * block_size
                        row_start = i * block_size
                        col_end = min(col_start + block_size, width)
                        row_end = min(row_start + block_size, height)

                        window = Window(col_start, row_start,
                                        col_end - col_start, row_end - row_start)

                                             
                        block_data = src.read(1, window=window)
                        block_mask = mask_src.read(1, window=window)

                                             
                        if debug_mode and debug_blocks_shown < max_debug_blocks:
                            if np.sum(block_mask == 1) > 0:                   
                                result_block, block_valid_pixels = process_pixels_debug(
                                    block_data, block_mask, lower_threshold, upper_threshold,
                                    nodata, block_idx=f"[{i},{j}]"
                                )
                                debug_blocks_shown += 1
                            else:
                                result_block, block_valid_pixels = process_pixels_fast(
                                    block_data, block_mask, lower_threshold, upper_threshold, nodata
                                )
                        else:
                                         
                            result_block, block_valid_pixels = process_pixels_fast(
                                block_data, block_mask, lower_threshold, upper_threshold, nodata
                            )

                                        
                        dst.write(result_block, window=window)

                              
                        processed_pixels += block_data.size
                        valid_pixels_found += block_valid_pixels

                                  
                        if pbar.n % 50 == 0:                  
                            elapsed = time.time() - start_processing_time
                            if elapsed > 0:
                                speed = pbar.n / elapsed
                                postfix_info = f"速度{speed:.1f}块/s 内存{get_memory_usage():.0f}MB 找到{valid_pixels_found:,}像素"
                                pbar.set_postfix_str(postfix_info)

                        pbar.update(1)

            
    print(f"\n🧹 清理临时文件...")
    try:
        os.remove(temp_mask_path)
        print(f"✅ 临时mask文件已删除")
    except:
        print(f"⚠️ 临时文件清理失败: {temp_mask_path}")

            
    end_time = time.time()
    total_time = end_time - start_time
    processing_time = end_time - start_processing_time

    print(f"\n🎉 处理完成!")
    print(f"📊 最终统计:")
    print(f"   ⏱️ 总耗时: {format_time(total_time)}")
    print(f"   ⏱️ 处理耗时: {format_time(processing_time)}")
    print(f"   📏 总像素数: {processed_pixels:,}")
    print(f"   ✅ 红色像素数 (整个TIF范围): {valid_pixels_found:,}")
    print(f"   📈 红色像素比例: {(valid_pixels_found / processed_pixels * 100):.6f}%")
    if processing_time > 0:
        print(f"   🚀 平均处理速度: {total_blocks / processing_time:.1f} 块/秒")
        print(f"   📊 像素处理速度: {processed_pixels / processing_time / 1000000:.1f} 百万像素/秒")
    print(f"   💾 峰值内存: {get_memory_usage():.1f}MB")
    print(f"   📁 输出文件: {output_path}")

          
    if valid_pixels_found == 0:
        print(f"\n⚠️ 警告: 没有找到符合条件的像素!")
        print(f"💡 可能的原因:")
        print(f"   1. 阈值范围太严格 (均值±2σ)")
        print(f"   2. 整个TIF图像的值都超出了SHP区域统计的范围")
        print(f"   3. 数据分布异常或有极值")
        print(f"   4. nodata值处理问题")
        print(f"💡 建议:")
        print(f"   1. 检查上述调试信息中的数据范围")
        print(f"   2. 尝试放宽阈值范围 (如均值±3σ)")
        print(f"   3. 检查TIF数据的整体分布")
    else:
        print(f"\n🎉 成功! 在整个TIF范围内找到了 {valid_pixels_found:,} 个符合条件的红色像素")
        print(f"💡 说明: 这些红色像素可能出现在SHP区域内外，只要值在阈值范围内")

    print("=" * 60)


     
if __name__ == "__main__":
          
    import argparse
    parser = argparse.ArgumentParser(description='Recovered reference-guided intensity-threshold implementation')
    parser.add_argument('--sar', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    tif_path, shp_path, output_path = args.sar, args.reference, args.output

    print("🚀 高性能地理数据处理脚本启动")
    print("=" * 60)

           
    required_packages = ['rasterio', 'geopandas', 'numpy', 'tqdm', 'psutil']
    optional_packages = ['numba']

    print("📦 检查依赖包...")
    missing_packages = []
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing_packages.append(pkg)

    if missing_packages:
        print(f"❌ 缺少必需依赖包: {', '.join(missing_packages)}")
        print(f"请运行: pip install {' '.join(missing_packages)}")
        exit(1)

    for pkg in optional_packages:
        try:
            __import__(pkg)
            print(f"✅ 可选包 {pkg} 已安装 - 将获得额外性能提升")
        except ImportError:
            print(f"💡 建议安装 {pkg} 获得更好性能: pip install {pkg}")

          
    print("\n📁 检查文件...")
    if not os.path.exists(tif_path):
        print(f"❌ TIF文件不存在: {tif_path}")
        exit(1)
    else:
        file_size_mb = os.path.getsize(tif_path) / 1024 / 1024
        print(f"✅ TIF文件存在 (大小: {file_size_mb:.1f}MB)")

    if not os.path.exists(shp_path):
        print(f"❌ SHP文件不存在: {shp_path}")
        exit(1)
    else:
        print(f"✅ SHP文件存在")

            
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 创建输出目录: {output_dir}")

          
    print(f"\n⚡ 性能优化特性:")
    print(f"   🎯 预计算全局mask - 避免重复栅格化")
    print(f"   📊 向量化数组操作 - 减少循环开销")
    print(f"   🧩 更大处理块 (2048×2048) - 减少I/O次数")
    print(f"   💾 LZW压缩 - 节省存储空间")
    if NUMBA_AVAILABLE:
        print(f"   🚀 Numba JIT编译 - 数值计算加速")
    print(f"   📁 临时文件清理 - 自动管理磁盘空间")

    print(f"\n🔄 处理逻辑:")
    print(f"   🎯 在SHP区域内计算统计值 (均值±2σ)")
    print(f"   📊 用这个阈值筛选整个TIF图像")
    print(f"   🔴 结果: 整个TIF中符合条件的像素都变红色")

    try:
                         
        process_tif_with_shp_optimized(tif_path, shp_path, output_path, block_size=2048, debug_mode=True)

                
        if os.path.exists(output_path):
            output_size_mb = os.path.getsize(output_path) / 1024 / 1024
            print(f"\n🎉 处理完全成功!")
            print(f"📁 输出文件: {output_path}")
            print(f"📏 输出文件大小: {output_size_mb:.1f}MB")
            print(f"💡 说明: 红色像素表示整个TIF范围内符合SHP区域统计阈值的像素")
            print(f"🎯 逻辑: 在SHP区域计算均值±2σ，然后筛选整个TIF图像")
        else:
            print("❌ 输出文件未生成")

    except Exception as e:
        print(f"\n❌ 处理过程中发生错误:")
        print(f"   错误信息: {e}")
        print(f"   内存使用: {get_memory_usage():.1f}MB")

        import traceback

        print(f"\n🔍 详细错误信息:")
        traceback.print_exc()

        print(f"\n💡 优化建议:")
        print(f"   1. 确保有足够的磁盘空间用于临时文件")
        print(f"   2. 考虑减小block_size参数 (如1024)")
        print(f"   3. 关闭其他占用内存的程序")
        print(f"   4. 检查TIF和SHP文件完整性")
        print(f"   5. 安装numba获得额外性能: pip install numba")

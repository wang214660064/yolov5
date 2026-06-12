import os
import sys
import argparse
import importlib
import pandas as pd
from openpyxl.styles import PatternFill
from openpyxl.utils import get_column_letter
from multiprocessing import Pool
import time

# 添加当前目录到路径，确保能导入同目录的模块
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

filter_duplicate_images = None


def ensure_dependencies():
    """检查并加载重复检测所需依赖。"""
    required_modules = {
        'Pillow': 'PIL',
        'imagehash': 'imagehash',
        'opencv-python': 'cv2',
        'scikit-image': 'skimage',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'openpyxl': 'openpyxl',
    }
    missing = []
    for package_name, module_name in required_modules.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        print("缺少以下依赖，无法执行重复检测和报告生成：")
        for package_name in missing:
            print(f"  - {package_name}")
        print("\n请先安装依赖，例如：")
        print("  pip install Pillow imagehash opencv-python scikit-image numpy pandas openpyxl")
        return False

    global filter_duplicate_images
    import filter_duplicate_images as duplicate_module
    filter_duplicate_images = duplicate_module
    return True


def parse_class_file(class_file_path):
    """解析 class.txt 文件"""
    class_mapping = {}
    
    with open(class_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # 解析格式: A:焊盘暗-引线脱离焊盘
            if ':' in line:
                code, name = line.split(':', 1)
                class_mapping[code.strip()] = {
                    'code': code.strip(),
                    'name': name.strip()
                }
    
    return class_mapping


def count_images_in_folder(folder_path, recursive=True):
    """
    统计文件夹中的图片数量
    
    参数：
        folder_path: 文件夹路径
        recursive: 是否递归统计子文件夹（默认True，包含子文件夹中的图片）
    
    返回：
        count: 图片总数
        all_files: 所有图片文件的路径列表
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
    count = 0
    all_files = []
    
    if recursive:
        # 递归遍历所有子文件夹（包含子文件夹中的图片）
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                if file_name.lower().endswith(image_extensions):
                    file_path = os.path.join(root, file_name)
                    count += 1
                    all_files.append(file_path)
    else:
        # 只统计当前文件夹
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path) and file_name.lower().endswith(image_extensions):
                count += 1
                all_files.append(file_path)
    
    return count, all_files


def compute_file_hash(file_path):
    """计算单个文件的 MD5 哈希值（用于多进程）"""
    try:
        return file_path, filter_duplicate_images.get_file_hash(file_path)
    except Exception as e:
        return file_path, None


def compute_phash(file_path):
    """计算单个文件的 pHash 值（用于多进程）"""
    try:
        return file_path, filter_duplicate_images.get_phash(file_path)
    except Exception as e:
        return file_path, None


def preload_hashes(results, num_processes=4):
    """
    预加载所有图片的哈希值（使用多进程并行化）
    
    参数：
        results: 文件夹统计结果列表
        num_processes: 进程数
    
    返回：
        md5_dict: {file_path: md5_hash}
        phash_dict: {file_path: phash}
    """
    all_files = []
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
    
    # 收集所有图片文件（递归包含子文件夹）
    for result in results:
        folder_path = result['folder_path']
        # 使用递归方式收集所有文件
        for root, dirs, files in os.walk(folder_path):
            for f in files:
                if f.lower().endswith(image_extensions):
                    all_files.append(os.path.join(root, f))
    
    total_files = len(all_files)
    print(f"\n预加载 {total_files} 个文件的哈希值...")
    
    # 使用多进程计算 MD5 哈希
    print(f"  [1/2] 计算 MD5 哈希值...")
    start_time = time.time()
    with Pool(processes=num_processes) as pool:
        md5_results = pool.map(compute_file_hash, all_files)
    
    md5_dict = {fp: h for fp, h in md5_results if h is not None}
    elapsed = time.time() - start_time
    print(f"        完成！耗时 {elapsed:.2f} 秒")
    
    # 使用多进程计算 pHash
    print(f"  [2/2] 计算 pHash 哈希值...")
    start_time = time.time()
    with Pool(processes=num_processes) as pool:
        phash_results = pool.map(compute_phash, all_files)
    
    phash_dict = {fp: h for fp, h in phash_results if h is not None}
    elapsed = time.time() - start_time
    print(f"        完成！耗时 {elapsed:.2f} 秒")
    
    return md5_dict, phash_dict


def detect_duplicates(folder_path):
    """
    检测文件夹中的重复图片
    
    返回：
        exact_dups: 完全重复组列表
        similar_dups: 相似重复对列表（已排除完全重复的图片）
    """
    # 检测完全重复（MD5哈希）
    exact_dups = filter_duplicate_images.find_exact_duplicates(folder_path)
    
    # 获取所有完全重复的文件路径
    exact_dup_files = set()
    for group in exact_dups:
        for file in group:
            exact_dup_files.add(file)
    
    # 检测视觉相似（pHash + SSIM）
    similar_dups = filter_duplicate_images.find_similar_duplicates(folder_path)
    
    # 过滤：从相似重复中排除完全重复的图片对
    # 完全重复肯定是相似的，所以不需要重复统计
    filtered_similar_dups = []
    for pair in similar_dups:
        file1, file2 = pair[0], pair[1]
        # 如果两个文件都不在完全重复列表中，才保留这个相似对
        if file1 not in exact_dup_files or file2 not in exact_dup_files:
            filtered_similar_dups.append(pair)
    
    print(f"    完全重复组: {len(exact_dups)}, 原始相似对: {len(similar_dups)}, 过滤后相似对: {len(filtered_similar_dups)}")
    
    return exact_dups, filtered_similar_dups


def count_images_in_data(data_dir, class_mapping):
    """统计 Data 目录下各类别的图片数量和重复情况（包含子文件夹）"""
    results = []
    
    for folder_name in sorted(class_mapping.keys()):
        folder_path = os.path.join(data_dir, folder_name)
        
        if not os.path.isdir(folder_path):
            print(f"跳过不存在的类别目录: {folder_name}")
            continue
        
        # 统计图片数量（递归包含子文件夹，如A分类下的"不明显"文件夹）
        image_count, all_files = count_images_in_folder(folder_path, recursive=True)
        
        class_info = class_mapping[folder_name]
        
        # 检测重复
        exact_dups, similar_dups = detect_duplicates(folder_path)
        
        # 将绝对路径转换为相对路径（相对于data_dir）
        exact_dups_rel = []
        for group in exact_dups:
            rel_group = [os.path.relpath(f, data_dir) for f in group]
            exact_dups_rel.append(rel_group)
        
        similar_dups_rel = []
        for pair in similar_dups:
            rel_pair = (os.path.relpath(pair[0], data_dir), 
                       os.path.relpath(pair[1], data_dir), 
                       pair[2], pair[3])
            similar_dups_rel.append(rel_pair)
        
        # 计算实际重复文件数（每组保留一个）
        exact_file_count = sum(len(g) - 1 for g in exact_dups)
        
        results.append({
            'folder_name': folder_name,
            'folder_path': folder_path,

            'class_name': class_info['name'],
            'image_count': image_count,
            'exact_dup_groups': len(exact_dups),
            'exact_dup_files': exact_file_count,
            'similar_dup_pairs': len(similar_dups),
            'exact_dup_groups_raw': exact_dups_rel,      # 保留相对路径数据
            'similar_dup_pairs_raw': similar_dups_rel    # 保留相对路径数据
        })
    
    return results


def compare_folders(args):
    """
    比对两个文件夹（用于多进程）
    
    参数：
        args: (small_folder, big_folder, md5_dict, phash_dict, base_dir)
    
    返回：
        (small_name, big_name, exact_count, similar_count, exact_pairs, similar_pairs)
    """
    small_folder, big_folder, md5_dict, phash_dict, base_dir = args
    
    small_path = small_folder['folder_path']
    small_name = small_folder['folder_name']
    big_path = big_folder['folder_path']
    big_name = big_folder['folder_name']
    
    # 收集两个文件夹的所有图片（递归包含子文件夹）
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
    small_files = []
    for root, dirs, files in os.walk(small_path):
        for f in files:
            if f.lower().endswith(image_extensions):
                small_files.append(os.path.join(root, f))
    
    big_files = []
    for root, dirs, files in os.walk(big_path):
        for f in files:
            if f.lower().endswith(image_extensions):
                big_files.append(os.path.join(root, f))
    
    # 检测完全重复（使用预加载的 MD5 哈希）
    exact_pairs = []
    from collections import defaultdict
    hash_groups = defaultdict(list)
    
    for file in small_files:
        md5_hash = md5_dict.get(file)
        if md5_hash:
            hash_groups[md5_hash].append(('small', file))
    
    for file in big_files:
        md5_hash = md5_dict.get(file)
        if md5_hash:
            hash_groups[md5_hash].append(('big', file))
    
    # 获取所有完全重复的文件
    exact_dup_files_set = set()
    for hash_val, files in hash_groups.items():
        if len(files) >= 2:
            small_file = None
            big_file = None
            for source, filepath in files:
                if source == 'small':
                    small_file = filepath
                else:
                    big_file = filepath
            if small_file and big_file:
                exact_dup_files_set.add(small_file)
                exact_dup_files_set.add(big_file)
                small_file_rel = os.path.relpath(small_file, base_dir)
                big_file_rel = os.path.relpath(big_file, base_dir)
                exact_pairs.append((small_name, big_name, small_file_rel, big_file_rel))
    
    # 检测视觉相似（使用预加载的 pHash）- 只对小文件夹进行，且排除完全重复的文件
    similar_pairs = []
    if len(small_files) > 0:
        for small_file in small_files:
            # 跳过完全重复的文件
            if small_file in exact_dup_files_set:
                continue
                
            phash1 = phash_dict.get(small_file)
            if phash1 is None:
                continue
            
            for big_file in big_files:
                # 跳过完全重复的文件
                if big_file in exact_dup_files_set:
                    continue
                    
                phash2 = phash_dict.get(big_file)
                if phash2 is None:
                    continue
                
                hamming_distance = phash1 - phash2
                if hamming_distance <= 4:
                    ssim_value = filter_duplicate_images.ssim_check(small_file, big_file)
                    if ssim_value >= 0.9:
                        small_file_rel = os.path.relpath(small_file, base_dir)
                        big_file_rel = os.path.relpath(big_file, base_dir)
                        similar_pairs.append((small_name, big_name, small_file_rel, big_file_rel, hamming_distance, ssim_value))
    
    return small_name, big_name, exact_pairs, similar_pairs


def find_cross_folder_duplicates(results, md5_dict, phash_dict):
    """
    跨文件夹重复检测：小文件夹和大文件夹比对（使用多进程并行化）
    返回：跨文件夹完全重复列表、相似重复列表、以及交叉矩阵
    """
    # 获取基础目录（用于相对路径转换）
    base_dir = os.path.dirname(results[0]['folder_path']) if results else ''
    
    # 按图片数量排序，小文件夹在前
    sorted_results = sorted(results, key=lambda x: x['image_count'])
    
    # 生成所有需要比对的文件夹对
    all_pairs = []
    for i, small_folder in enumerate(sorted_results):
        for j in range(i + 1, len(sorted_results)):
            big_folder = sorted_results[j]
            all_pairs.append((small_folder, big_folder, md5_dict, phash_dict, base_dir))
    
    total_pairs = len(all_pairs)
    print(f"\n跨文件夹比对开始...")
    print(f"  总比对对数: {total_pairs}")
    
    # 使用多进程并行比对
    cross_exact_dups = []
    cross_similar_dups = []
    completed = [0]  # 使用列表以便在闭包中修改
    
    def update_progress(result):
        completed[0] += 1
        progress = (completed[0] / total_pairs) * 100
        print(f"\r  进度: [{completed[0]}/{total_pairs}] {progress:.1f}%", end='', flush=True)
        return result
    
    start_time = time.time()
    with Pool(processes=4) as pool:
        # 使用 imap_unordered 获取结果并显示进度
        for small_name, big_name, exact_pairs, similar_pairs in pool.imap_unordered(compare_folders, all_pairs):
            cross_exact_dups.extend(exact_pairs)
            cross_similar_dups.extend(similar_pairs)
            completed[0] += 1
            progress = (completed[0] / total_pairs) * 100
            print(f"\r  进度: [{completed[0]}/{total_pairs}] {progress:.1f}%", end='', flush=True)
    
    elapsed = time.time() - start_time
    print(f"\n  完成！耗时 {elapsed:.2f} 秒")
    
    # 创建交叉矩阵
    folder_names = [r['folder_name'] for r in results]
    exact_matrix = pd.DataFrame(0, index=folder_names, columns=folder_names)
    similar_matrix = pd.DataFrame(0, index=folder_names, columns=folder_names)
    
    # 统计每对文件夹的重复数量
    exact_counts = {}
    similar_counts = {}
    
    for pair in cross_exact_dups:
        key = (pair[0], pair[1])
        exact_counts[key] = exact_counts.get(key, 0) + 1
    
    for pair in cross_similar_dups:
        key = (pair[0], pair[1])
        similar_counts[key] = similar_counts.get(key, 0) + 1
    
    # 填充交叉矩阵
    for (small, big), count in exact_counts.items():
        exact_matrix.loc[small, big] = count
        exact_matrix.loc[big, small] = count
    
    for (small, big), count in similar_counts.items():
        similar_matrix.loc[small, big] = count
        similar_matrix.loc[big, small] = count
    
    print(f"  跨文件夹完全重复: {len(cross_exact_dups)} 对")
    print(f"  跨文件夹相似重复: {len(cross_similar_dups)} 对")
    
    return cross_exact_dups, cross_similar_dups, exact_matrix, similar_matrix


def generate_excel(results, cross_exact_dups, cross_similar_dups, exact_matrix, similar_matrix, output_path):
    """生成统计 Excel 文件"""
    from openpyxl.styles import Font, Alignment, Border, Side, PatternFill
    
    stat_columns = [
        '类别编码', '类别名称', '图片数量', '完全重复组数', '完全重复文件数', '相似重复对数'
    ]
    stats_df = pd.DataFrame([{
        '类别编码': item['folder_name'],
        '类别名称': item['class_name'],
        '图片数量': item['image_count'],
        '完全重复组数': item['exact_dup_groups'],
        '完全重复文件数': item['exact_dup_files'],
        '相似重复对数': item['similar_dup_pairs'],
    } for item in results], columns=stat_columns)
    
    # 添加总计行
    total_row = pd.DataFrame({
        '类别编码': ['总计'],
        '类别名称': [''],
        '图片数量': [stats_df['图片数量'].sum()],
        '完全重复组数': [stats_df['完全重复组数'].sum()],
        '完全重复文件数': [stats_df['完全重复文件数'].sum()],
        '相似重复对数': [stats_df['相似重复对数'].sum()],
    }, columns=stat_columns)
    stats_df = pd.concat([stats_df, total_row], ignore_index=True)
    
    # 准备颜色列表（用于标记重复组）
    colors = [
        'FFFFCC',  # 浅黄色
        'CCFFCC',  # 浅绿色
        'CCFFFF',  # 浅蓝色
        'FFCCCC',  # 浅红色
        'FFCCFF',  # 浅紫色
        'CCFFEE',  # 浅青色
        'EEFFCC',  # 浅黄绿色
        'FFEECC',  # 浅橙色
        'EECCFF',  # 浅紫罗兰
        'CCEEFF',  # 浅天蓝色
    ]
    
    # 定义样式
    header_font = Font(bold=True, color='FFFFFF')
    header_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
    total_fill = PatternFill(start_color='FFC000', end_color='FFC000', fill_type='solid')
    total_font = Font(bold=True, color='000000')
    align_center = Alignment(horizontal='center', vertical='center')
    align_left = Alignment(horizontal='left', vertical='center')
    thin_border = Border(left=Side(style='thin'), right=Side(style='thin'), 
                         top=Side(style='thin'), bottom=Side(style='thin'))

    def set_widths(worksheet, widths):
        for col_idx, width in enumerate(widths, start=1):
            worksheet.column_dimensions[get_column_letter(col_idx)].width = width

    def style_header(worksheet, column_count):
        for col in range(1, column_count + 1):
            cell = worksheet.cell(row=1, column=col)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = align_center
            cell.border = thin_border

    def style_body(worksheet, row_count, column_count, left_columns=(), color_column=None):
        for row in range(2, row_count + 2):
            fill = None
            if color_column:
                color_idx = worksheet.cell(row=row, column=color_column).value
                if color_idx is not None:
                    fill = PatternFill(start_color=colors[color_idx], end_color=colors[color_idx], fill_type='solid')
            for col in range(1, column_count + 1):
                cell = worksheet.cell(row=row, column=col)
                if fill:
                    cell.fill = fill
                cell.alignment = align_left if col in left_columns else align_center
                cell.border = thin_border

    def write_sheet(writer, sheet_name, data, columns, widths, left_columns=(), color_column=None):
        df = pd.DataFrame(data, columns=columns)
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        worksheet = writer.sheets[sheet_name]
        set_widths(worksheet, widths)
        style_header(worksheet, len(columns))
        style_body(worksheet, len(df), len(columns), left_columns=left_columns, color_column=color_column)
        return df, worksheet
    
    # 写入 Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        stats_df.to_excel(writer, index=False, sheet_name='统计结果')
        worksheet = writer.sheets['统计结果']
        set_widths(worksheet, [12, 28, 12, 14, 16, 14])
        style_header(worksheet, len(stat_columns))
        for row in range(2, len(stats_df) + 2):
            for col in range(1, len(stat_columns) + 1):
                cell = worksheet.cell(row=row, column=col)
                cell.alignment = align_left if col == 2 else align_center
                cell.border = thin_border
                if row == len(stats_df) + 1:
                    cell.fill = total_fill
                    cell.font = total_font
        
        # 第二张表：完全重复文件明细（带颜色标记）
        exact_data = []
        group_id = 0
        
        for result in results:
            for group in result['exact_dup_groups_raw']:
                color_index = group_id % len(colors)
                for file in group:
                    exact_data.append({
                        '类别': result['folder_name'],
                        '类别名称': result['class_name'],
                        '文件路径': file,
                        '重复组ID': group_id,
                        '颜色索引': color_index
                    })
                group_id += 1
        
        exact_columns = ['类别', '类别名称', '文件路径', '重复组ID', '颜色索引']
        exact_df, worksheet2 = write_sheet(
            writer, '完全重复文件', exact_data, exact_columns, [8, 25, 60, 10, 10],
            left_columns=(3,), color_column=5
        )
        
        # 第三张表：相似重复文件明细（带颜色标记）
        similar_data = []
        similar_group_id = 0
        
        for result in results:
            for pair in result['similar_dup_pairs_raw']:
                color_index = similar_group_id % len(colors)
                similar_data.append({
                    '类别': result['folder_name'],
                    '类别名称': result['class_name'],
                    '文件1': pair[0],
                    '文件2': pair[1],
                    '汉明距离': pair[2],
                    'SSIM值': pair[3],
                    '重复组ID': similar_group_id,
                    '颜色索引': color_index
                })
                similar_group_id += 1
        
        similar_columns = ['类别', '类别名称', '文件1', '文件2', '汉明距离', 'SSIM值', '重复组ID', '颜色索引']
        similar_df, worksheet3 = write_sheet(
            writer, '相似重复文件', similar_data, similar_columns, [8, 25, 55, 55, 12, 12, 10, 10],
            left_columns=(3, 4), color_column=8
        )
        
        # 第四张表：跨文件夹完全重复
        cross_exact_data = []
        for item in cross_exact_dups:
            cross_exact_data.append({
                '小文件夹': item[0],
                '大文件夹': item[1],
                '小文件夹文件': item[2],
                '大文件夹文件': item[3]
            })
        
        cross_exact_columns = ['小文件夹', '大文件夹', '小文件夹文件', '大文件夹文件']
        cross_exact_df, worksheet4 = write_sheet(
            writer, '跨文件夹完全重复', cross_exact_data, cross_exact_columns, [10, 10, 55, 55],
            left_columns=(3, 4)
        )
        
        # 第五张表：跨文件夹相似重复
        cross_similar_data = []
        for item in cross_similar_dups:
            cross_similar_data.append({
                '小文件夹': item[0],
                '大文件夹': item[1],
                '小文件夹文件': item[2],
                '大文件夹文件': item[3],
                '汉明距离': item[4],
                'SSIM值': item[5]
            })
        
        cross_similar_columns = ['小文件夹', '大文件夹', '小文件夹文件', '大文件夹文件', '汉明距离', 'SSIM值']
        cross_similar_df, worksheet5 = write_sheet(
            writer, '跨文件夹相似重复', cross_similar_data, cross_similar_columns, [10, 10, 50, 50, 12, 12],
            left_columns=(3, 4)
        )
        
        # 第六张表：跨文件夹完全重复交叉矩阵
        exact_matrix.to_excel(writer, index=True, sheet_name='完全重复交叉矩阵')
        worksheet6 = writer.sheets['完全重复交叉矩阵']
        
        # 设置列宽
        worksheet6.column_dimensions['A'].width = 10
        for i, col in enumerate(exact_matrix.columns, start=2):
            worksheet6.column_dimensions[get_column_letter(i)].width = 10
        
        # 设置表头样式
        for col in range(1, len(exact_matrix.columns) + 2):
            cell = worksheet6.cell(row=1, column=col)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = align_center
            cell.border = thin_border
        
        # 设置索引列样式
        for row in range(2, len(exact_matrix) + 2):
            cell = worksheet6.cell(row=row, column=1)
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color='D9D9D9', end_color='D9D9D9', fill_type='solid')
            cell.alignment = align_center
            cell.border = thin_border
        
        # 设置数据单元格样式
        for row in range(2, len(exact_matrix) + 2):
            for col in range(2, len(exact_matrix.columns) + 2):
                cell = worksheet6.cell(row=row, column=col)
                cell.alignment = align_center
                cell.border = thin_border
                # 对角线上的单元格（同一文件夹）设置灰色背景
                if row == col:
                    cell.fill = PatternFill(start_color='EFEFEF', end_color='EFEFEF', fill_type='solid')
        
        # 第七张表：跨文件夹相似重复交叉矩阵
        similar_matrix.to_excel(writer, index=True, sheet_name='相似重复交叉矩阵')
        worksheet7 = writer.sheets['相似重复交叉矩阵']
        
        # 设置列宽
        worksheet7.column_dimensions['A'].width = 10
        for i, col in enumerate(similar_matrix.columns, start=2):
            worksheet7.column_dimensions[get_column_letter(i)].width = 10
        
        # 设置表头样式
        for col in range(1, len(similar_matrix.columns) + 2):
            cell = worksheet7.cell(row=1, column=col)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = align_center
            cell.border = thin_border
        
        # 设置索引列样式
        for row in range(2, len(similar_matrix) + 2):
            cell = worksheet7.cell(row=row, column=1)
            cell.font = Font(bold=True)
            cell.fill = PatternFill(start_color='D9D9D9', end_color='D9D9D9', fill_type='solid')
            cell.alignment = align_center
            cell.border = thin_border
        
        # 设置数据单元格样式
        for row in range(2, len(similar_matrix) + 2):
            for col in range(2, len(similar_matrix.columns) + 2):
                cell = worksheet7.cell(row=row, column=col)
                cell.alignment = align_center
                cell.border = thin_border
                if row == col:
                    cell.fill = PatternFill(start_color='EFEFEF', end_color='EFEFEF', fill_type='solid')
    
    print(f"\nExcel 文件已生成: {output_path}")


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    
    # 获取路径
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    class_file = os.path.join(base_dir, 'class.txt')
    default_data_dir = r"E:\Desktop\MAC-WIN\04 OpenCV\98_Practice\3DPrinterNozzleInspection\dataset_4"

    parser = argparse.ArgumentParser(description='焊盘缺陷检测图片统计工具')
    parser.add_argument('--data-dir', default=default_data_dir, help='要统计的数据目录，默认 Data/00_data_original')
    parser.add_argument('--output', default=None, help='Excel 报告输出路径，默认写到数据目录下的 图片统计报告.xlsx')
    parser.add_argument('--processes', type=int, default=4, help='哈希预加载的进程数，默认 4')
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    output_path = os.path.abspath(args.output) if args.output else os.path.join(data_dir, '图片统计报告.xlsx')
    
    print("=" * 60)
    print("焊盘缺陷检测图片统计工具")
    print("=" * 60)
    print(f"类别文件: {class_file}")
    print(f"数据目录: {data_dir}")
    print(f"输出文件: {output_path}")
    print(f"注意: 统计包含子文件夹（如A分类下的'不明显'文件夹）")

    if not os.path.isdir(data_dir):
        print(f"\n数据目录不存在: {data_dir}")
        return

    if not ensure_dependencies():
        return
    
    # 解析类别文件
    class_mapping = parse_class_file(class_file)
    print(f"\n已解析 {len(class_mapping)} 个类别")
    
    # 统计图片数量和重复检测（包含子文件夹）
    print("\n[阶段1/3] 统计图片数量和单文件夹重复检测...")
    start_time = time.time()
    results = count_images_in_data(data_dir, class_mapping)
    elapsed = time.time() - start_time
    print(f"        完成！耗时 {elapsed:.2f} 秒")
    
    if not results:
        print("\n未找到任何图片文件夹")
        return
    
    # 打印统计结果
    print("\n统计结果:")
    print("-" * 90)
    print(f"{'文件夹':<12} {'类别名称':<20} {'总数':<6} {'重复组':<8} {'重复文件':<8} {'相似对':<8}")
    print("-" * 90)
    
    total_images = 0
    total_exact_groups = 0
    total_exact_files = 0
    total_similar = 0
    
    for result in results:
        print(f"{result['folder_name']:<12} {result['class_name']:<20} "
              f"{result['image_count']:<6} {result['exact_dup_groups']:<8} "
              f"{result['exact_dup_files']:<8} {result['similar_dup_pairs']:<8}")
        total_images += result['image_count']
        total_exact_groups += result['exact_dup_groups']
        total_exact_files += result['exact_dup_files']
        total_similar += result['similar_dup_pairs']
    
    print("-" * 90)
    print(f"{'总计':<12} {'':<6} {'':<20} {total_images:<6} {total_exact_groups:<8} {total_exact_files:<8} {total_similar:<8}")
    
    # 预加载哈希值（使用多进程）
    print("\n[阶段2/3] 预加载所有图片哈希值（多进程）...")
    md5_dict, phash_dict = preload_hashes(results, num_processes=args.processes)
    
    # 跨文件夹重复检测（使用多进程并行化）
    print("\n[阶段3/3] 跨文件夹重复检测（多进程并行）...")
    cross_exact_dups, cross_similar_dups, exact_matrix, similar_matrix = find_cross_folder_duplicates(results, md5_dict, phash_dict)
    
    # 生成 Excel
    print("\n生成 Excel 报告...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    generate_excel(results, cross_exact_dups, cross_similar_dups, exact_matrix, similar_matrix, output_path)
    
    print("\n" + "=" * 60)
    print("统计完成！")
    print("注意：相似重复已排除完全重复的图片（完全重复肯定是相似的）")
    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
简化版文件名分组脚本
根据文件名前8位数字进行分组
"""

import os
import json
import re
from collections import defaultdict


def group_images_by_filename(pic_dir="./Pic"):
    """根据文件名中的数字前缀对图片进行分组"""
    groups = defaultdict(list)
    
    # 遍历所有子目录
    for root, dirs, files in os.walk(pic_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                # 提取文件名中的数字前缀（至少8位）
                match = re.search(r'(\d{8,})', file)
                if match:
                    # 取前8位作为产品ID
                    product_id = match.group(1)[:8]
                    # 获取相对路径
                    rel_path = os.path.relpath(os.path.join(root, file), pic_dir)
                    rel_path = rel_path.replace("\\", "/")  # 统一路径分隔符
                    groups[product_id].append(rel_path)
    
    # 转换为所需的格式
    similarity_annotations = []
    for product_id, image_list in groups.items():
        if len(image_list) >= 1:  # 至少有一张图片才创建条目
            # 将第一个图片作为query_image，其余作为relevant_images
            query_image = image_list[0]
            relevant_images = image_list[1:] if len(image_list) > 1 else []
            
            # 从路径中提取类别信息
            category = os.path.basename(os.path.dirname(query_image)) if os.path.dirname(query_image) else "unknown"
            
            similarity_annotations.append({
                "query_image": query_image,
                "relevant_images": relevant_images,
                "category": category,
                "product_id": product_id
            })
    
    return similarity_annotations


def main():
    print("开始按文件名规则分组...")
    
    if not os.path.exists("./Pic"):
        print("错误: 未找到 ./Pic 目录")
        return
    
    annotations = group_images_by_filename()
    
    print(f"分组完成，共生成 {len(annotations)} 个分组")
    
    # 保存结果
    with open("./similarity_annotations.json", 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    print("结果已保存至 similarity_annotations.json")
    

if __name__ == "__main__":
    main()
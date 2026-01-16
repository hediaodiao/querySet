#!/usr/bin/env python3
"""
三阶段流水线主控制器
实现完整的 "文件名规则初筛 → 模型视觉校验 → 人工最终审核" 流程
"""

import subprocess
import sys
import os
import json
from pathlib import Path


def run_command(cmd, desc="执行命令"):
    """运行命令并显示进度"""
    print(f"\n🔍 {desc}")
    print(f"   命令: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"   ✅ {desc} 完成")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"   ❌ {desc} 失败")
        print(f"   错误: {e.stderr}")
        return False, e.stderr


def check_prerequisites():
    """检查前置条件"""
    print("📋 检查前置条件...")
    
    # 检查必要的Python包
    required_packages = ['torch', 'torchvision', 'PIL', 'sklearn', 'numpy']
    missing_packages = []
    
    for pkg in required_packages:
        try:
            if pkg == 'PIL':
                import PIL
            elif pkg == 'sklearn':
                import sklearn
            elif pkg == 'torch':
                import torch
            elif pkg == 'torchvision':
                import torchvision
            elif pkg == 'numpy':
                import numpy
        except ImportError:
            missing_packages.append(pkg)
    
    if missing_packages:
        print(f"❌ 缺少必要包: {missing_packages}")
        print("请运行: pip install torch torchvision pillow scikit-learn numpy")
        return False
    
    # 检查图片目录
    if not os.path.exists("./Pic"):
        print("❌ 未找到 ./Pic 目录")
        return False
    
    print("✅ 前置条件检查通过")
    return True


def stage_1_filename_grouping():
    """第一阶段：文件名规则分组"""
    print("\n" + "="*60)
    print("🚀 第一阶段：文件名规则分组")
    print("="*60)
    
    if os.path.exists("./similarity_annotations.json"):
        print("✅ 检测到已存在的分组结果文件")
        with open("./similarity_annotations.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"   已有 {len(data)} 个分组条目")
        return True
    
    print("⚠️ 未找到分组结果文件，需要运行 main2.py")
    print("💡 注意: 请确保您有 main2.py 文件来进行文件名规则分组")
    
    # 如果没有main2.py，我们创建一个简化版本
    if not os.path.exists("./main2.py"):
        print("💡 创建简化版文件名分组脚本...")
        create_simple_grouping_script()
    
    success, _ = run_command(["python", "main2.py"], "运行文件名分组")
    return success


def create_simple_grouping_script():
    """创建简化版的文件名分组脚本"""
    script_content = '''
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
    """根据文件名前8位数字对图片进行分组"""
    groups = defaultdict(list)
    
    # 遍历所有子目录
    for root, dirs, files in os.walk(pic_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                # 提取文件名中的数字前缀（前8位）
                match = re.search(r'(\\d{8,})', file)
                if match:
                    # 取前8位作为产品ID
                    product_id = match.group(1)[:8]
                    # 获取相对路径
                    rel_path = os.path.relpath(os.path.join(root, file), pic_dir)
                    rel_path = rel_path.replace("\\\\", "/")  # 统一路径分隔符
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
    annotations = group_images_by_filename()
    
    print(f"分组完成，共生成 {len(annotations)} 个分组")
    
    # 保存结果
    with open("./similarity_annotations.json", 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    
    print("结果已保存至 similarity_annotations.json")
    

if __name__ == "__main__":
    main()
'''
    
    with open("./main2.py", 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    print("✅ 简化版分组脚本创建完成")


def stage_2_model_verification():
    """第二阶段：模型视觉校验"""
    print("\n" + "="*60)
    print("🔍 第二阶段：模型视觉校验")
    print("="*60)
    
    if not os.path.exists("./similarity_annotations.json"):
        print("❌ 未找到基础分组文件，无法进行视觉校验")
        return False
    
    success, _ = run_command(["python", "openclip_only_verification.py"], "运行模型视觉校验")
    return success


def stage_3_manual_review():
    """第三阶段：人工审核"""
    print("\n" + "="*60)
    print("👥 第三阶段：人工审核")
    print("="*60)
    
    # 检查是否已生成审核报告
    verification_report_exists = os.path.exists("./similarity_annotations_verification_report.json")
    merge_report_exists = os.path.exists("./similarity_annotations_review_report.json")
    
    if verification_report_exists or merge_report_exists:
        print("✅ 检测到校验报告，可以开始人工审核")
        print("\n💡 运行以下命令之一进行审核：")
        print("   python group_review_tool.py     # 基础审核")
        print("   python review_tool_optimized.py # 优化审核（如有合并建议）")
        return True
    else:
        print("⚠️ 未找到校验报告，将直接使用基础分组结果进行审核")
        if os.path.exists("./similarity_annotations.json"):
            print("✅ 基础分组文件存在，可以直接进行审核")
            return True
        else:
            print("❌ 无法找到可用的分组文件进行审核")
            return False


def generate_final_report():
    """生成最终报告"""
    print("\n" + "="*60)
    print("📊 生成最终报告")
    print("="*60)
    
    report = {
        "pipeline_status": "completed",
        "stages_completed": [],
        "files_generated": [],
        "summary": {}
    }
    
    # 检查生成的文件
    files_to_check = [
        "similarity_annotations.json",
        "similarity_annotations_verification_report.json",
        "similarity_annotations_initial.json",
        "similarity_annotations_final.json",
        "similarity_annotations_review_report.json"
    ]
    
    for file in files_to_check:
        if os.path.exists(file):
            size = os.path.getsize(file)
            report["files_generated"].append({
                "name": file,
                "size_bytes": size,
                "exists": True
            })
            
            # 如果是JSON文件，尝试加载统计信息
            if file.endswith('.json'):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            report["summary"][file] = f"{len(data)} 个项目"
                        elif isinstance(data, dict) and "verified_annotations" in data:
                            report["summary"][file] = f"{len(data['verified_annotations'])} 个验证条目"
                except:
                    pass
        else:
            report["files_generated"].append({
                "name": file,
                "exists": False
            })
    
    # 保存报告
    with open("pipeline_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("✅ 最终报告已生成: pipeline_report.json")
    
    # 打印摘要
    print("\n📋 流水线摘要:")
    for file_info in report["files_generated"]:
        status = "✅" if file_info["exists"] else "❌"
        print(f"   {status} {file_info['name']}")
        if file_info['name'] in report['summary']:
            print(f"      └─ {report['summary'][file_info['name']]}")
    
    return True


def main():
    """主函数：执行完整流水线"""
    print("🎨 图片分组三阶段流水线")
    print("   阶段1: 文件名规则初筛")
    print("   阶段2: 模型视觉校验") 
    print("   阶段3: 人工最终审核")
    print()
    
    # 检查前置条件
    if not check_prerequisites():
        print("\n❌ 前置条件检查失败，退出")
        return
    
    # 执行三个阶段
    stages_success = []
    
    # 阶段1：文件名分组
    success = stage_1_filename_grouping()
    stages_success.append(("Stage 1 - Filename Grouping", success))
    
    if not success:
        print("\n❌ 第一阶段失败，流水线终止")
        return
    
    # 阶段2：模型视觉校验
    success = stage_2_model_verification()
    stages_success.append(("Stage 2 - Model Verification", success))
    
    if not success:
        print("\n⚠️  第二阶段失败，但仍可进行人工审核")
    
    # 阶段3：人工审核
    success = stage_3_manual_review()
    stages_success.append(("Stage 3 - Manual Review", success))
    
    # 生成最终报告
    generate_final_report()
    
    # 总结
    print("\n" + "="*60)
    print("🏁 流水线执行完成")
    print("="*60)
    
    for stage, success in stages_success:
        status = "✅" if success else "❌"
        print(f"   {status} {stage}")
    
    completed_count = sum(1 for _, success in stages_success if success)
    print(f"\n📈 完成率: {completed_count}/{len(stages_success)} 阶段")


if __name__ == "__main__":
    main()
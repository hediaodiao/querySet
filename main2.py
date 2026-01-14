#!/usr/bin/env python3
"""
高效加载本地预训练模型
"""

import os
import json
import re
from collections import defaultdict
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ImageSimilarityGenerator:
    def __init__(self, base_path, model_name='resnet50'):
        """
        初始化相似图片生成器
        
        Args:
            base_path: 图片根目录路径，包含buildingBlock和watergun子文件夹
            model_name: 使用的预训练模型名称
        """
        self.base_path = base_path
        self.categories = ['buildingBlock', 'watergun']
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        
        # 初始化模型
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = self._load_model(model_name)
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
        
    def _load_model(self, model_name):
        """加载预训练模型"""
        if model_name == 'resnet50':
            model = models.resnet50(pretrained=True)
            # 移除最后的分类层
            model = torch.nn.Sequential(*list(model.children())[:-1])
        elif model_name == 'efficientnet_b0':
            from torchvision.models import efficientnet_b0
            # 加载本地EfficientNet-B0模型
            model = self._load_custom_efficientnet()
        else:
            raise ValueError(f"不支持的模型: {model_name}")
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def _load_custom_efficientnet(self):
        """加载本地EfficientNet-B0模型"""
        from torchvision.models import efficientnet_b0
        
        # 创建模型架构
        model = efficientnet_b0(pretrained=False)
        
        # 加载本地权重
        checkpoint = torch.load('model_cache/efficientnet-b0-355c32eb.pth', 
                               map_location=self.device, 
                               weights_only=False)
        
        # 检查权重字典的键名，可能需要映射到EfficientNet的期望键名
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        
        # 检查有多少键匹配
        print(f"匹配的键数量: {len(pretrained_dict)}/{len(model_dict)}")
        
        # 更新模型权重
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        
        # 移除分类层
        model.classifier = torch.nn.Identity()
        
        return model
    
    def extract_product_id(self, filename):
        """
        从文件名中提取产品ID（前8位数字）
        """
        match = re.search(r'(\d+)', filename)
        if match:
            digits = match.group(1)
            return digits[:8] if len(digits) >= 8 else digits
        return None
    
    def extract_image_features(self, image_path):
        """
        提取图片特征向量
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(image_tensor)
                features = features.squeeze().cpu().numpy()
            
            return features
        except Exception as e:
            print(f"处理图片 {image_path} 时出错: {e}")
            return None
    
    def group_by_filename(self):
        """
        基于文件名规则进行初步分组
        """
        file_based_groups = defaultdict(lambda: defaultdict(list))
        
        for category in self.categories:
            category_path = os.path.join(self.base_path, category)
            if not os.path.exists(category_path):
                print(f"警告: 目录 {category_path} 不存在")
                continue
                
            for filename in os.listdir(category_path):
                if filename.lower().endswith(self.image_extensions):
                    product_id = self.extract_product_id(filename)
                    if product_id:
                        image_path = os.path.join(category, filename)
                        file_based_groups[product_id][category].append(image_path)
        
        print(f"基于文件名规则分组完成，共发现 {len(file_based_groups)} 个产品ID")
        return file_based_groups
    
    def validate_with_model(self, file_based_groups, similarity_threshold=0.7):
        """
        使用模型视觉特征验证和调整分组（组内相似度分析）
        """
        adjusted_groups = defaultdict(lambda: defaultdict(list))
        
        for product_id, categories in file_based_groups.items():
            for category, image_paths in categories.items():
                if len(image_paths) <= 1:
                    adjusted_groups[product_id][category] = image_paths
                    continue
                
                # 提取所有图片的特征
                features_dict = {}
                valid_paths = []
                
                for img_path in image_paths:
                    full_path = os.path.join(self.base_path, img_path)
                    features = self.extract_image_features(full_path)
                    if features is not None:
                        features_dict[img_path] = features
                        valid_paths.append(img_path)
                
                if len(valid_paths) <= 1:
                    adjusted_groups[product_id][category] = image_paths
                    continue
                
                # 计算特征相似度矩阵
                feature_matrix = np.array([features_dict[path] for path in valid_paths])
                
                # 使用DBSCAN聚类进行视觉验证
                clustering = DBSCAN(eps=0.5, min_samples=1).fit(feature_matrix)
                labels = clustering.labels_
                
                # 根据聚类结果重新分组
                clusters = defaultdict(list)
                for i, label in enumerate(labels):
                    clusters[label].append(valid_paths[i])
                
                if len(clusters) == 1:
                    adjusted_groups[product_id][category] = image_paths
                else:
                    # 视觉特征发现需要拆分分组
                    for cluster_id, cluster_paths in clusters.items():
                        new_product_id = f"{product_id}_cluster_{cluster_id}"
                        adjusted_groups[new_product_id][category] = cluster_paths
                    
                    print(f"产品ID {product_id} 根据视觉特征拆分为 {len(clusters)} 个组")
        
        return adjusted_groups

    def discover_and_suggest_merges(self, adjusted_groups, merge_threshold=0.85):
        """
        【新增功能】发现并建议合并视觉上高度相似的不同分组
        返回: 合并建议列表供人工审核
        """
        print("开始分析组间相似度，寻找可合并的分组...")
        
        # 为每个分组计算平均特征向量作为代表
        group_representatives = {}
        
        for product_id, categories in adjusted_groups.items():
            all_group_features = []
            for category, image_paths in categories.items():
                for img_path in image_paths:
                    full_path = os.path.join(self.base_path, img_path)
                    features = self.extract_image_features(full_path)
                    if features is not None:
                        all_group_features.append(features)
            
            if all_group_features:
                group_rep = np.mean(all_group_features, axis=0)
                group_representatives[product_id] = group_rep
        
        # 计算不同组代表向量之间的余弦相似度
        product_ids = list(group_representatives.keys())
        merge_suggestions = []
        
        for i in range(len(product_ids)):
            id_i = product_ids[i]
            vec_i = group_representatives[id_i]
            
            for j in range(i + 1, len(product_ids)):
                id_j = product_ids[j]
                vec_j = group_representatives[id_j]
                
                # 计算余弦相似度
                similarity = self.cosine_similarity(vec_i, vec_j)
                
                # 如果相似度高于合并阈值，则建议合并
                if similarity > merge_threshold:
                    merge_suggestions.append({
                        'group_a': id_i,
                        'group_b': id_j,
                        'similarity_score': float(similarity),
                        'recommended_action': 'MERGE'
                    })
                    print(f"建议合并: {id_i} 和 {id_j}, 相似度: {similarity:.4f}")
        
        print(f"组间相似度分析完成，共产生 {len(merge_suggestions)} 条合并建议。")
        return merge_suggestions

    def cosine_similarity(self, a, b):
        """计算两个向量之间的余弦相似度"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def generate_json_structure(self, final_groups):
        """
        生成最终的JSON结构
        """
        json_data = []
        
        for product_id, categories in final_groups.items():
            for category, image_paths in categories.items():
                if len(image_paths) < 2:
                    continue
                
                # 为每组生成所有可能的查询对
                for i, query_image in enumerate(image_paths):
                    relevant_images = [path for path in image_paths if path != query_image]
                    
                    entry = {
                        "query_image": query_image,
                        "relevant_images": relevant_images,
                        "category": category,
                        "product_id": product_id
                    }
                    json_data.append(entry)
        
        return json_data

    def save_human_review_template(self, json_data, merge_suggestions, output_path):
        """
        生成便于人工审核的模板文件（增强版：包含合并建议）
        """
        # 按产品ID分组，便于人工审核
        review_data = defaultdict(list)
        
        for entry in json_data:
            key = (entry['product_id'], entry['category'])
            review_data[key].append(entry)
        
        # 生成增强的审核报告
        review_report = {
            "summary": {
                "total_products": len(review_data),
                "total_query_pairs": len(json_data),
                "merge_suggestions_count": len(merge_suggestions),
                "categories": self.categories
            },
            "merge_suggestions": merge_suggestions,
            "merge_notes": "请人工审核以下分组对。如果确认是同一商品的不同变体（如不同颜色、尺寸），请执行合并操作。",
            "products_need_review": []
        }
        
        for (product_id, category), entries in review_data.items():
            product_info = {
                "product_id": product_id,
                "category": category,
                "image_count": len(set([e['query_image'] for e in entries] + 
                                      [img for e in entries for img in e['relevant_images']])),
                "query_pairs_count": len(entries),
                "sample_images": entries[0]['relevant_images'][:3] if entries else []
            }
            review_report["products_need_review"].append(product_info)
        
        # 保存审核报告
        review_path = output_path.replace('.json', '_review_report.json')
        with open(review_path, 'w', encoding='utf-8') as f:
            json.dump(review_report, f, indent=2, ensure_ascii=False)
        
        print(f"人工审核报告已生成: {review_path}")

    def run_pipeline(self, output_json_path, enable_model_validation=True, enable_merge_detection=True):
        """
        运行完整的数据生成流程（增强版）
        """
        print("=== 开始生成相似图片标注数据集 ===")
        
        # 1. 基于文件名规则分组
        print("步骤1: 基于文件名规则进行初步分组...")
        file_based_groups = self.group_by_filename()
        
        # 2. 使用模型进行视觉验证（可选）
        if enable_model_validation:
            print("步骤2: 使用模型进行视觉特征验证...")
            final_groups = self.validate_with_model(file_based_groups)
        else:
            final_groups = file_based_groups
        
        # 3. 【新增】组间相似度分析与合并建议
        merge_suggestions = []
        if enable_merge_detection and enable_model_validation:
            print("步骤3: 组间相似度分析，寻找需要合并的分组...")
            merge_suggestions = self.discover_and_suggest_merges(final_groups)
        
        # 4. 生成JSON结构
        print("步骤4: 生成JSON数据结构...")
        json_data = self.generate_json_structure(final_groups)
        
        # 5. 保存结果
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        # 6. 生成人工审核报告（包含合并建议）
        self.save_human_review_template(json_data, merge_suggestions, output_json_path)
        
        print(f"=== 流程完成 ===")
        print(f"生成的标注数据集已保存至: {output_json_path}")
        print(f"共生成 {len(json_data)} 个查询对")
        print(f"涉及 {len(final_groups)} 个产品ID")
        print(f"发现 {len(merge_suggestions)} 个需要人工审核的合并建议")

def main():
    # 配置参数
    BASE_PATH = "picTest"  # 指向图片根目录路径
    OUTPUT_JSON = "similarity_annotations.json"
    
    # 创建生成器实例，使用efficientnet_b0模型
    generator = ImageSimilarityGenerator(BASE_PATH, model_name='efficientnet_b0')
    
    # 运行完整流程
    generator.run_pipeline(OUTPUT_JSON, 
                          enable_model_validation=True,  # 启用模型验证
                          enable_merge_detection=True)   # 启用合并检测

if __name__ == "__main__":
    main()
import json
import os
import numpy as np
import torch
import open_clip
from PIL import Image
from collections import defaultdict
import time
from datetime import datetime
import pickle
from pathlib import Path
import torchvision.transforms as transforms
import cv2

# 配置选项：是否重新生成图片特征缓存
# 设置为 True 时会清除现有缓存并重新生成特征；设置为 False 时使用现有缓存
REBUILD_FEATURE_CACHE = False

# 配置选项：是否只使用本地模型，不进行在线下载
# 设置为 True 时只使用本地模型，如果本地模型不存在则终止程序
# 设置为 False 时允许从网络下载模型（首次运行需要网络连接）
USE_LOCAL_MODEL_ONLY = False


class OpenCLIPFeatureExtractor:
    """
    使用OpenCLIP模型提取图片特征
    支持缓存机制和批量处理
    """
    
    def __init__(self, cache_dir="./model_cache/features", device=None):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 如果REBUILD_FEATURE_CACHE为True，则清除现有缓存
        if REBUILD_FEATURE_CACHE:
            self.clear_cache()
        
        # 设置设备
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🎯 使用设备: {self.device}")
        
        # 加载OpenCLIP模型
        model_name = "timm/vit_base_patch32_clip_224.laion2b_e16"
        
        print("🔄 加载OpenCLIP模型...")
        
        try:
            # 检查是否有本地模型文件
            local_model_path = "./model_cache/timm/vit_base_patch32_clip_224.laion2b_e16"
            
            # 检查本地模型文件是否存在（检查特定的模型权重文件）
            config_path = os.path.join(local_model_path, "open_clip_config.json")
            safetensors_path = os.path.join(local_model_path, "open_clip_model.safetensors")
            bin_path = os.path.join(local_model_path, "pytorch_model.bin")
            model_path = os.path.join(local_model_path, "model.safetensors")
            
            local_model_files_exist = (
                os.path.exists(config_path) and
                (os.path.exists(safetensors_path) or 
                 os.path.exists(bin_path) or
                 os.path.exists(model_path))
            )
            
            if local_model_files_exist:
                print(f"📁 发现本地模型文件: {local_model_path}")
                
                # 加载本地模型 - 使用正确的格式
                self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                    model_name='ViT-B-32',
                    pretrained='laion2b_e16',
                    cache_dir='./model_cache'
                )
                print("✅ 从本地加载OpenCLIP模型成功")
            else:
                if USE_LOCAL_MODEL_ONLY:
                    print(f"❌ 未找到本地模型文件: {local_model_path}")
                    print("❌ 由于USE_LOCAL_MODEL_ONLY=True，项目终止")
                    raise FileNotFoundError(f"本地模型文件不存在: {local_model_path}")
                else:
                    # 尝试下载模型
                    print(f"🌐 正在下载OpenCLIP模型: {model_name}")
                    self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                        model_name='ViT-B-32',
                        pretrained='laion2b_e16',
                        cache_dir='./model_cache'
                    )
                    print("✅ 下载并加载OpenCLIP模型成功")
                
        except Exception as e:
            print(f"❌ 加载OpenCLIP模型失败: {e}")
            print("💡 请确保已安装open_clip_torch: pip install open_clip_torch")
            raise
        
        # 移动模型到设备
        self.model = self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        
    def get_feature_cache_path(self, image_path):
        """获取特征缓存文件路径"""
        # 使用图像路径的哈希值作为缓存文件名
        image_hash = hash(image_path) % (10**16)  # 限制哈希长度
        return self.cache_dir / f"{image_hash}.pkl"
    
    def load_cached_feature(self, image_path):
        """从缓存加载特征"""
        cache_path = self.get_feature_cache_path(image_path)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
        return None
    
    def save_feature_to_cache(self, image_path, feature):
        """保存特征到缓存"""
        cache_path = self.get_feature_cache_path(image_path)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(feature, f)
        except Exception as e:
            print(f"⚠️  保存特征缓存失败: {e}")
    
    def clear_cache(self):
        """清除特征缓存"""
        import shutil
        if self.cache_dir.exists():
            print(f"🗑️  清除特征缓存目录: {self.cache_dir}")
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print("✅ 特征缓存已清除")
        else:
            print(f"ℹ️  缓存目录不存在: {self.cache_dir}")
    
    def preprocess_image(self, image_path):
        """预处理单张图片"""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.preprocess(image)
        except Exception as e:
            print(f"⚠️  预处理图片失败 {image_path}: {e}")
            return None
    
    def extract_features_batch(self, image_paths, batch_size=8):
        """批量提取特征"""
        features = {}
        
        # 分批处理
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            
            # 加载并预处理批次图片
            batch_images = []
            valid_paths = []
            
            for img_path in batch_paths:
                full_path = os.path.join("./Pic", img_path)
                if os.path.exists(full_path):
                    # 如果设置了强制重建缓存，则跳过加载缓存
                    if hasattr(self, '_force_rebuild_cache') and self._force_rebuild_cache:
                        cached_feature = None
                    else:
                        cached_feature = self.load_cached_feature(full_path)
                    
                    if cached_feature is not None:
                        features[img_path] = cached_feature
                        continue
                    
                    preprocessed_img = self.preprocess_image(full_path)
                    if preprocessed_img is not None:
                        batch_images.append(preprocessed_img)
                        valid_paths.append(img_path)
            
            # 批量处理未缓存的图片
            if batch_images:
                # 转换为tensor并移动到设备
                batch_tensor = torch.stack(batch_images).to(self.device)
                
                # 提取特征
                with torch.no_grad():
                    if self.device.type == 'cuda':
                        with torch.cuda.amp.autocast():
                            image_features = self.model.encode_image(batch_tensor)
                    else:
                        image_features = self.model.encode_image(batch_tensor)
                    image_features /= image_features.norm(dim=-1, keepdim=True)  # 归一化
                
                # 转换为numpy数组并保存到缓存
                image_features_np = image_features.cpu().numpy()
                
                for j, img_path in enumerate(valid_paths):
                    feature = image_features_np[j]
                    features[img_path] = feature
                    
                    # 保存到缓存
                    full_path = os.path.join("./Pic", img_path)
                    self.save_feature_to_cache(full_path, feature)
        
        return features


class OptimizedModelVisualVerifier:
    """
    优化的模型视觉验证器
    使用OpenCLIP模型、缓存机制和批量处理
    """
    
    def __init__(self, base_path="./Pic", cache_dir="./model_cache/features", batch_size=8):
        self.base_path = base_path
        self.batch_size = batch_size
        
        # 初始化特征提取器
        self.feature_extractor = OpenCLIPFeatureExtractor(cache_dir=cache_dir)
        
        print("💡 初始化优化模型视觉验证器")
        
    def calculate_similarity(self, feat1, feat2):
        """计算余弦相似度"""
        if feat1 is None or feat2 is None:
            return 0.0
        # 计算余弦相似度
        similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
        return max(0.0, similarity)  # 确保非负
    
    def verify_similarity_annotations(self, annotations_file="similarity_annotations.json", 
                                     output_path="similarity_annotations_verification_report.json"):
        """验证相似度标注文件 - 优化版"""
        print("🔍 开始验证相似度标注文件...")
        
        # 读取标注文件
        with open(annotations_file, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        print(f"📊 总共 {len(annotations)} 个产品组待验证")
        
        # 按产品ID分组
        product_groups = defaultdict(list)
        for entry in annotations:
            product_id = entry.get('product_id', 'unknown')
            product_groups[product_id].append(entry)
        
        product_ids = list(product_groups.keys())
        print(f"📦 发现 {len(product_ids)} 个不同的产品ID")
        
        verified_annotations = []
        confirmed_groups = []
        split_suggestions = []
        merge_suggestions = []
        
        # 首先提取所有图片的特征（批量处理）
        print("\n🔄 批量提取所有图片特征...")
        all_image_paths = set()
        for entries in product_groups.values():
            for entry in entries:
                all_image_paths.add(entry['query_image'])
                all_image_paths.update(entry['relevant_images'])
        
        all_image_paths = list(all_image_paths)
        print(f"总共需要处理 {len(all_image_paths)} 张唯一图片")
        
        # 批量提取特征
        all_features = self.feature_extractor.extract_features_batch(
            all_image_paths, 
            batch_size=self.batch_size
        )
        print(f"✅ 特征提取完成，共 {len(all_features)} 个特征")
        
        # 1. 检查组内一致性
        print("\n开始检查组内一致性...")
        for idx, (product_id, entries) in enumerate(product_groups.items()):
            print(f"  处理产品组 {idx+1}/{len(product_ids)}: {product_id}")
            
            # 计算query_image与relevant_images之间的相似度
            all_intra_similarities = []
            low_similarity_images = []  # 存储低相似度的图片
            
            for entry in entries:
                query_img = entry['query_image']
                
                # 计算query_image与每个relevant_image的相似度
                entry_similarities = []
                for rel_img in entry['relevant_images']:
                    if query_img in all_features and rel_img in all_features:
                        sim = self.calculate_similarity(all_features[query_img], all_features[rel_img])
                        entry_similarities.append(sim)
                        
                        # 如果相似度低于阈值，标记为需要人工审核
                        if sim < 0.5:  # 设定阈值为0.5
                            low_similarity_images.append({
                                'query_image': query_img,
                                'relevant_image': rel_img,
                                'similarity': sim
                            })
                
                # 计算该entry的平均相似度
                avg_entry_similarity = np.mean(entry_similarities) if entry_similarities else 1.0
                all_intra_similarities.append(avg_entry_similarity)
            
            # 计算整个产品的平均相似度
            avg_intra_similarity = np.mean(all_intra_similarities) if all_intra_similarities else 1.0
            print(f"  组内平均相似度: {avg_intra_similarity:.3f}")
            
            # 根据相似度决定处理方式
            if avg_intra_similarity >= 0.85:  # 高相似度，确认分组
                for entry in entries:
                    entry['verification_status'] = 'confirmed_high_sim'
                    entry['intra_similarity'] = avg_intra_similarity
                    entry['low_similarity_images'] = []  # 没有低相似度图片
                    verified_annotations.append(entry)
                confirmed_groups.append(product_id)
                print(f"  → 确认分组 {product_id} (相似度: {avg_intra_similarity:.3f})")
            elif avg_intra_similarity < 0.5:  # 低相似度，建议拆分
                split_suggestions.append({
                    'product_id': product_id,
                    'avg_similarity': avg_intra_similarity,
                    'image_count': sum([len(entry['relevant_images']) + 1 for entry in entries]),  # 总图片数
                    'images': low_similarity_images,  # 包含低相似度图片信息
                    'low_similarity_details': low_similarity_images
                })
                # 暂时保留原条目，等待人工处理
                for entry in entries:
                    entry['verification_status'] = 'needs_split'
                    entry['intra_similarity'] = avg_intra_similarity
                    entry['low_similarity_images'] = low_similarity_images  # 添加低相似度图片信息
                    verified_annotations.append(entry)
                print(f"  → 建议拆分 {product_id} (相似度: {avg_intra_similarity:.3f}), 低相似度图片数: {len(low_similarity_images)}")
            else:
                # 中等相似度，保持原样，人工审核
                for entry in entries:
                    entry['verification_status'] = 'needs_review'
                    entry['intra_similarity'] = avg_intra_similarity
                    entry['low_similarity_images'] = low_similarity_images  # 添加低相似度图片信息
                    verified_annotations.append(entry)
                print(f"  → 需要人工审核 {product_id} (相似度: {avg_intra_similarity:.3f}), 低相似度图片数: {len(low_similarity_images)}")
        
        # 2. 检查组间合并建议（仅比较query图片）
        print("\n开始检查组间合并建议（仅比较query图片）...")
        total_comparisons = len(product_ids) * (len(product_ids) - 1) // 2
        print(f"总共需要进行 {total_comparisons} 次组间相似度比较")
        
        product_ids = list(product_groups.keys())
        processed_count = 0
        start_time = time.time()
        
        for i in range(len(product_ids)):
            for j in range(i + 1, len(product_ids)):
                pid1, pid2 = product_ids[i], product_ids[j]
                
                # 显示进度
                processed_count += 1
                if processed_count % max(1, total_comparisons // 20) == 0:  # 每5%显示一次进度
                    elapsed = time.time() - start_time
                    eta = (elapsed / processed_count) * (total_comparisons - processed_count)
                    print(f"  进度: {processed_count}/{total_comparisons} "
                          f"({processed_count/total_comparisons*100:.1f}%) "
                          f"[耗时: {elapsed:.1f}s, 预计剩余: {eta:.1f}s]")
                
                # 获取两个产品的query图片（第一个图片）
                query_img1 = product_groups[pid1][0]['query_image']
                query_img2 = product_groups[pid2][0]['query_image']
                
                # 比较query图片的相似度
                if query_img1 in all_features and query_img2 in all_features:
                    similarity = self.calculate_similarity(all_features[query_img1], all_features[query_img2])
                    
                    # 如果组间相似度高，建议合并
                    if similarity >= 0.75:  # 高相似度，建议合并
                        merge_suggestions.append({
                            'group_a': pid1,
                            'group_b': pid2,
                            'similarity_score': similarity,
                            'group_a_size': len([img for entry in product_groups[pid1] 
                                               for img in [entry['query_image']] + entry['relevant_images']]),
                            'group_b_size': len([img for entry in product_groups[pid2] 
                                               for img in [entry['query_image']] + entry['relevant_images']]),
                            'group_a_query': query_img1,
                            'group_b_query': query_img2
                        })
        
        print(f"\n✅ 完成所有相似度验证")
        print(f"📈 结果统计:")
        print(f"   - 确认的分组: {len(confirmed_groups)}")
        print(f"   - 建议拆分: {len(split_suggestions)}")
        print(f"   - 建议合并: {len(merge_suggestions)}")
        print(f"   - 需要人工审核: {len([x for x in verified_annotations if x.get('verification_status') == 'needs_review'])}")
        
        # 转换numpy数据类型为Python原生类型以支持JSON序列化
        def convert_numpy_types(obj):
            """递归转换numpy数据类型为Python原生类型"""
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # 保存验证报告
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_groups_processed': len(product_groups),
            'confirmed_groups_count': len(confirmed_groups),
            'split_suggestions_count': len(split_suggestions),
            'merge_suggestions_count': len(merge_suggestions),
            'verified_annotations': verified_annotations,
            'split_suggestions': split_suggestions,
            'merge_suggestions': merge_suggestions,
            'processing_summary': {
                'groups_confirmed': confirmed_groups,
                'groups_needing_split': [s['product_id'] for s in split_suggestions],
                'group_pairs_for_merge': [(m['group_a'], m['group_b']) for m in merge_suggestions]
            }
        }
        
        # 转换numpy类型以支持JSON序列化
        report = convert_numpy_types(report)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # 同时保存更新后的基础标注文件
        base_annotations_path = output_path.replace('_verification_report.json', '.json')
        with open(base_annotations_path, 'w', encoding='utf-8') as f:
            json.dump(convert_numpy_types(verified_annotations), f, indent=2, ensure_ascii=False)
        
        print(f"更新后的标注文件已保存至: {base_annotations_path}")
        
        return verified_annotations, report


def main():
    """主函数：执行模型视觉验证"""
    print("🚀 优化版模型视觉验证器")
    print("="*50)
    
    # 创建验证器实例
    verifier = OptimizedModelVisualVerifier(
        cache_dir="./model_cache/features",
        batch_size=8  # 可根据GPU内存调整
    )
    
    # 执行验证（使用现有文件）
    annotations_file = "similarity_annotations.json"
    if os.path.exists(annotations_file):
        print(f"\n📋 找到标注文件: {annotations_file}")
        try:
            verified_annotations, report = verifier.verify_similarity_annotations(
                annotations_file=annotations_file
            )
            print(f"\n✅ 验证完成！")
        except FileNotFoundError:
            print(f"⚠️  未找到 {annotations_file}，跳过实际验证")
            print("   提示: 先运行 main2.py 生成 similarity_annotations.json")
    else:
        print(f"⚠️  未找到 {annotations_file}，跳过实际验证")
        print("   提示: 先运行 main2.py 生成 similarity_annotations.json")


if __name__ == "__main__":
    main()
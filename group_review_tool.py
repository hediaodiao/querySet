#!/usr/bin/env python3
"""
按组图片审核工具 - 支持按组展示图片，判断某张图片是否属于某个组
"""

import os
import json
import random
from pathlib import Path
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import numpy as np


# 配置选项已移至 model_verification.py 文件中
# 请在 model_verification.py 中修改 REBUILD_FEATURE_CACHE 和 USE_LOCAL_MODEL_ONLY

app = FastAPI(title="按组图片审核工具")

# 挂载静态文件目录（用于显示图片）
app.mount("/static", StaticFiles(directory="./Pic"), name="static")

# 设置模板目录
templates = Jinja2Templates(directory="templates")


class GroupReviewManager:
    def __init__(self, similarity_data_path, base_path="./Pic"):
        self.base_path = base_path
        self.similarity_data_path = similarity_data_path
        self.load_similarity_data()
        self.initialize_review_state()
        # 初始化特征提取器
        try:
            from model_verification import OpenCLIPFeatureExtractor
            # 配置变量已在 model_verification.py 中定义
            self.feature_extractor = OpenCLIPFeatureExtractor(cache_dir="./model_cache/features")
        except ImportError:
            print("⚠️ 无法导入模型验证模块，相似度计算功能将不可用")
            self.feature_extractor = None
        # 加载验证报告
        self.verification_report = {
            "merge_suggestions": [],
            "split_suggestions": [],
            "confirmed_groups": []
        }
        self.load_verification_report()
    
    def load_similarity_data(self):
        """加载相似图片标注数据"""
        with open(self.similarity_data_path, 'r', encoding='utf-8') as f:
            self.similarity_data = json.load(f)
        
        # 按产品ID和类别重新组织数据
        self.groups_by_product = {}
        for entry in self.similarity_data:
            product_id = entry['product_id']
            category = entry['category']
            
            if product_id not in self.groups_by_product:
                self.groups_by_product[product_id] = {}
            
            if category not in self.groups_by_product[product_id]:
                self.groups_by_product[product_id][category] = {
                    'images': set(),
                    'query_pairs': []
                }
            
            # 添加所有图片到组中
            self.groups_by_product[product_id][category]['images'].add(entry['query_image'])
            for rel_img in entry['relevant_images']:
                self.groups_by_product[product_id][category]['images'].add(rel_img)
            
            self.groups_by_product[product_id][category]['query_pairs'].append(entry)
        
        print(f"✓ 加载了 {len(self.groups_by_product)} 个产品分组")
    
    def load_review_state(self):
        """加载审核状态"""
        state_file = self.similarity_data_path.replace('.json', '_group_review_state.json')
        if os.path.exists(state_file):
            with open(state_file, 'r', encoding='utf-8') as f:
                loaded_state = json.load(f)
                self.review_state.update(loaded_state)
        else:
            # 如果没有状态文件，使用默认状态
            self.review_state = {
                "processed_groups": [],
                "group_decisions": {},
                "current_review_index": 0,
                "undo_stack": []
            }
    
    def initialize_review_state(self):
        """初始化审核状态"""
        self.review_state = {
            "processed_groups": [],
            "group_decisions": {},
            "current_review_index": 0,
            "undo_stack": []  # 添加撤销栈
        }
        
        # 加载已有的审核状态（如果有）
        state_file = self.similarity_data_path.replace('.json', '_group_review_state.json')
        if os.path.exists(state_file):
            with open(state_file, 'r', encoding='utf-8') as f:
                loaded_state = json.load(f)
                self.review_state.update(loaded_state)
    
    def save_review_state(self):
        """保存审核状态"""
        state_file = self.similarity_data_path.replace('.json', '_group_review_state.json')
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(self.review_state, f, indent=2, ensure_ascii=False)
    
    def save_similarity_data_without_backup(self):
        """仅保存相似度数据到JSON文件（不进行备份）- 用于初始化时"""
        with open(self.similarity_data_path, 'w', encoding='utf-8') as f:
            json.dump(self.similarity_data, f, ensure_ascii=False, indent=2)
    
    def save_initial_similarity_data_backup(self):
        """保存初始相似度数据作为备份（仅在第一次加载时调用）"""
        import shutil
        import os
        # 检查是否已有初始数据备份
        initial_backup_path = self.similarity_data_path.replace('.json', '_initial.json')
        if not os.path.exists(initial_backup_path) and os.path.exists(self.similarity_data_path):
            shutil.copy2(self.similarity_data_path, initial_backup_path)
            print(f"✓ 已保存初始数据备份到: {initial_backup_path}")
        return initial_backup_path

    def save_similarity_data(self):
        """保存相似度数据到JSON文件（覆盖原始文件）"""
        # 首先保存初始数据备份（仅首次）
        self.save_initial_similarity_data_backup()
        
        # 保存到原始文件（会被覆盖）
        with open(self.similarity_data_path, 'w', encoding='utf-8') as f:
            json.dump(self.similarity_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 已保存当前工作进度到: {self.similarity_data_path}")
    
    def get_all_groups(self):
        """获取所有分组列表"""
        groups = []
        for product_id, categories in self.groups_by_product.items():
            for category, group_data in categories.items():
                groups.append({
                    'product_id': product_id,
                    'category': category,
                    'image_count': len(group_data['images']),
                    'images': list(group_data['images'])
                })
        return groups
    
    def get_group_candidates(self, group_id):
        """获取特定组的候选图片列表"""
        candidates = []
        for product_id, categories in self.groups_by_product.items():
            if product_id != group_id:  # 排除当前组
                for category, group_data in categories.items():
                    candidates.extend(list(group_data['images']))
        
        # 去重并返回
        return list(set(candidates))
    
    def get_next_candidate_for_group(self):
        """获取下一个待审核的组和候选图片"""
        groups = self.get_all_groups()
        
        # 优先处理当前正在进行但未完成的组（即还有剩余候选图片的组）
        for group in groups:
            group_id = group['product_id']
            if group_id not in self.review_state.get("processed_groups", []):
                # 获取当前组的剩余候选图片
                all_candidates = self.get_group_candidates(group_id)
                
                # 过滤掉已经处理过的候选图片
                processed_for_group = self.review_state.get("group_decisions", {}).get(group_id, [])
                processed_images = [dec.get("candidate_image") for dec in processed_for_group]
                
                remaining_candidates = [img for img in all_candidates if img not in processed_images]
                
                if remaining_candidates:
                    # 如果当前组还有剩余候选图片，优先返回当前组
                    candidate_image = random.choice(remaining_candidates)
                    return group_id, candidate_image, group['category']
        
        # 如果当前没有正在进行的组有剩余图片，则寻找新的组
        for group in groups:
            group_id = group['product_id']
            if group_id not in self.review_state.get("processed_groups", []):
                # 获取新组的所有候选图片
                all_candidates = self.get_group_candidates(group_id)
                
                # 过滤掉已经处理过的候选图片
                processed_for_group = self.review_state.get("group_decisions", {}).get(group_id, [])
                processed_images = [dec.get("candidate_image") for dec in processed_for_group]
                
                remaining_candidates = [img for img in all_candidates if img not in processed_images]
                
                if remaining_candidates:
                    candidate_image = random.choice(remaining_candidates)
                    return group_id, candidate_image, group['category']
        
        # 如果所有组都处理完了，返回None
        return None
    
    def get_group_candidates_for_review(self, group_id, num_candidates=10):
        """获取指定组用于审核的候选图片列表"""
        all_candidates = self.get_group_candidates(group_id)
        
        # 过滤掉已经处理过的候选图片
        processed_for_group = self.review_state.get("group_decisions", {}).get(group_id, [])
        processed_images = [dec.get("candidate_image") for dec in processed_for_group]
        
        remaining_candidates = [img for img in all_candidates if img not in processed_images]
        
        # 返回指定数量的候选图片，如果没有足够的图片则返回全部
        return remaining_candidates[:num_candidates]
    
    def get_group_images(self, group_id):
        """获取指定组的所有图片"""
        group_images = set()
        for entry in self.similarity_data:
            if entry['product_id'] == group_id:
                group_images.add(entry['query_image'])
                group_images.update(entry['relevant_images'])
        return list(group_images)
    
    def process_group_decision(self, group_id, candidate_image, decision):
        """处理组决策并更新JSON文件"""
        decision_record = {
            "group_id": group_id,
            "candidate_image": candidate_image,
            "decision": decision,
            "timestamp": str(Path(self.similarity_data_path).stat().st_mtime)
        }
        
        # 记录决策
        if group_id not in self.review_state["group_decisions"]:
            self.review_state["group_decisions"][group_id] = []
        
        # 存储撤销信息以便后续撤销操作
        undo_info = {
            "group_id": group_id,
            "candidate_image": candidate_image,
            "decision": decision,
            "previous_state": self._get_current_group_state(group_id, candidate_image)
        }
        
        self.review_state["group_decisions"][group_id].append(decision_record)
        
        # 保存撤销信息
        self.review_state["undo_stack"].append(undo_info)
        
        # 根据决策更新JSON数据
        if decision == "include":
            print(f"开始处理包含决策 - 组ID: {group_id}, 候选图片: {candidate_image}")
            try:
                # 将候选图片添加到当前组的所有相关图片中
                # 先找出当前组的所有相关图片条目
                group_entries = [entry for entry in self.similarity_data if entry['product_id'] == group_id]
                print(f"找到 {len(group_entries)} 个组条目")
                
                # 第一步：将候选图片添加到当前组的所有相关图片列表中
                for entry in group_entries:
                    print(f"处理条目: {entry['query_image']}, 当前相关图片数量: {len(entry['relevant_images'])}")
                    if candidate_image not in entry['relevant_images']:
                        entry['relevant_images'].append(candidate_image)
                        print(f"  添加 {candidate_image} 到 {entry['query_image']} 的相关图片列表")
                
                # 第二步：更新该组的其他图片的相关图片列表（确保相互引用的一致性）
                # 使用副本避免在遍历过程中修改列表
                for entry in group_entries:
                    # 创建相关图片列表的副本
                    rel_images_copy = list(entry['relevant_images'])
                    print(f"处理相互引用一致性，条目: {entry['query_image']}, 相关图片副本长度: {len(rel_images_copy)}")
                    
                    for rel_img in rel_images_copy:
                        if rel_img != candidate_image:
                            for other_entry in group_entries:
                                if other_entry['query_image'] == rel_img:
                                    if candidate_image not in other_entry['relevant_images']:
                                        other_entry['relevant_images'].append(candidate_image)
                                        print(f"  添加 {candidate_image} 到 {rel_img} 的相关图片列表")
                
                # 更新内部组数据结构
                for category, group_data in self.groups_by_product[group_id].items():
                    group_data['images'].add(candidate_image)
                    print(f"  更新内部组数据结构，添加 {candidate_image}")
                
                print(f"✓ 将图片 {candidate_image} 加入到组 {group_id}")
            except Exception as e:
                print(f"处理包含决策时发生错误: {str(e)}")
                import traceback
                print(f"详细错误信息: {traceback.format_exc()}")
                raise
        
        elif decision == "exclude":
            print(f"开始处理排除决策 - 组ID: {group_id}, 候选图片: {candidate_image}")
            try:
                # 从当前组的所有相关图片中移除候选图片
                # 先找出当前组的所有相关图片条目
                group_entries = [entry for entry in self.similarity_data if entry['product_id'] == group_id]
                print(f"找到 {len(group_entries)} 个组条目")
                
                # 首先处理该组的其他图片的相关图片列表
                for entry in group_entries:
                    if entry['query_image'] != candidate_image:
                        # 从其他图片的相关图片列表中移除候选图片
                        if candidate_image in entry['relevant_images']:
                            entry['relevant_images'].remove(candidate_image)
                            print(f"  从 {entry['query_image']} 的相关图片列表中移除 {candidate_image}")
                
                # 然后处理候选图片自身的相关图片列表（如果它属于该组）
                for entry in group_entries:
                    if entry['query_image'] == candidate_image:
                        # 清空候选图片的相关图片列表（因为它被移出了该组）
                        entry['relevant_images'] = []
                        print(f"  清空 {candidate_image} 的相关图片列表")
                
                # 更新内部组数据结构
                for category, group_data in self.groups_by_product[group_id].items():
                    if candidate_image in group_data['images']:
                        group_data['images'].remove(candidate_image)
                        print(f"  从内部组数据结构中移除 {candidate_image}")
                
                print(f"✓ 将图片 {candidate_image} 从组 {group_id} 移除")
            except Exception as e:
                print(f"处理排除决策时发生错误: {str(e)}")
                import traceback
                print(f"详细错误信息: {traceback.format_exc()}")
                raise
        
        # 保存更新后的JSON文件
        print("开始保存更新后的JSON文件")
        self.save_similarity_data()
        print("JSON文件保存完成")
        
        # 如果处理了足够多的决策，标记该组为已处理
        if len(self.review_state["group_decisions"][group_id]) >= 3:  # 每个组处理3个决策
            if group_id not in self.review_state["processed_groups"]:
                self.review_state["processed_groups"].append(group_id)
                print(f"组 {group_id} 标记为已处理")
        
        self.save_review_state()
        print("审核状态保存完成")
    
    def _get_current_group_state(self, group_id, candidate_image):
        """获取当前组状态用于撤销操作"""
        # 获取当前组的所有条目
        group_entries = [entry for entry in self.similarity_data if entry['product_id'] == group_id]
        
        # 返回当前状态下每个条目的相关信息
        state = {}
        for entry in group_entries:
            state[entry['query_image']] = {
                'relevant_images': entry['relevant_images'][:],  # 创建副本
                'is_query_image': entry['query_image'] == candidate_image
            }
        
        return state

    def load_verification_report(self):
        """加载模型验证报告"""
        self.verification_report = {
            "merge_suggestions": [],
            "split_suggestions": []
        }
        
        verification_report_path = "./similarity_annotations_verification_report.json"
        if os.path.exists(verification_report_path):
            with open(verification_report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
                
            self.verification_report["merge_suggestions"] = report.get("merge_suggestions", [])
            self.verification_report["split_suggestions"] = report.get("split_suggestions", [])
            self.verification_report["confirmed_groups"] = report.get("verified_annotations", [])
            
            print(f"✓ 加载了验证报告: {len(self.verification_report['merge_suggestions'])}个合并建议, {len(self.verification_report['split_suggestions'])}个拆分建议")
        else:
            print("⚠️ 未找到验证报告文件")
    
    def get_merge_suggestions_for_group(self, group_id):
        """获取指定组的合并建议"""
        suggestions = []
        for suggestion in self.verification_report["merge_suggestions"]:
            if suggestion["group_a"] == group_id or suggestion["group_b"] == group_id:
                suggestions.append(suggestion)
        return suggestions
    
    def get_split_suggestions_for_group(self, group_id):
        """获取指定组的拆分建议"""
        suggestions = []
        for suggestion in self.verification_report["split_suggestions"]:
            if suggestion["product_id"] == group_id:  # 注意：这里使用product_id匹配
                # 为拆分建议计算内部图片间的相似度
                detailed_suggestion = suggestion.copy()
                if 'images' in suggestion and self.feature_extractor:
                    # 计算建议拆分的图片与组内query图片的相似度
                    group_query_images = set()
                    for entry in self.similarity_data:
                        if entry['product_id'] == group_id:
                            group_query_images.add(entry['query_image'])
                    
                    if group_query_images:
                        # 计算拆分建议中的图片与query图片的相似度
                        split_images = suggestion.get('images', [])
                        split_image_similarities = {}
                        
                        for split_img in split_images:
                            # 计算该图片与组内所有query图片的相似度
                            max_similarity = 0.0
                            for query_img in group_query_images:
                                if split_img in group_query_images:
                                    # 如果拆分图片本身就是query图片，跳过
                                    continue
                                
                                # 提取特征并计算相似度
                                images_to_process = [split_img, query_img]
                                features = self.feature_extractor.extract_features_batch(images_to_process)
                                
                                if split_img in features and query_img in features:
                                    similarity = self.calculate_similarity(features[split_img], features[query_img])
                                    max_similarity = max(max_similarity, similarity)
                            
                            split_image_similarities[split_img] = max_similarity
                        
                        detailed_suggestion['split_image_similarities'] = split_image_similarities
                
                suggestions.append(detailed_suggestion)
        return suggestions
    
    def get_group_verification_status(self, group_id):
        """获取组的验证状态"""
        # 检查是否在确认的组中
        for entry in self.verification_report["confirmed_groups"]:
            if entry['product_id'] == group_id and entry.get('verification_status') == 'confirmed_high_sim':
                return 'confirmed_high_sim', entry.get('intra_similarity', 1.0)
        
        # 检查是否在拆分建议中
        for suggestion in self.verification_report["split_suggestions"]:
            if suggestion["product_id"] == group_id:
                return 'needs_split', suggestion.get('avg_similarity', 0.0)
        
        # 检查是否需要人工审核
        for entry in self.verification_report["confirmed_groups"]:
            if entry['product_id'] == group_id and entry.get('verification_status') == 'needs_review':
                return 'needs_review', entry.get('intra_similarity', 0.5)
        
        # 检查是否需要审核低相似度图片
        for entry in self.verification_report["confirmed_groups"]:
            if entry['product_id'] == group_id and entry.get('verification_status') in ['needs_split', 'needs_review']:
                low_sim_imgs = entry.get('low_similarity_images', [])
                if low_sim_imgs:
                    return 'has_low_similarity_images', entry.get('intra_similarity', 0.5)
        
        return 'unverified', 0.0
    
    def get_top_similar_images(self, query_image, top_k=20):
        """获取与query_image最相似的top_k张图片"""
        if not self.feature_extractor:
            return []
        
        # 获取所有图片文件
        all_image_paths = []
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    # 获取相对路径
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.base_path)
                    # 使用 / 作为路径分隔符以兼容web访问
                    rel_path = rel_path.replace("\\", "/")
                    all_image_paths.append(rel_path)
        
        # 提取query图片的特征
        query_features = self.feature_extractor.extract_features_batch([query_image.replace("./Pic/", "")])
        if query_image.replace("./Pic/", "") not in query_features:
            return []
        
        query_feat = query_features[query_image.replace("./Pic/", "")]
        
        # 提取所有其他图片的特征
        all_features = self.feature_extractor.extract_features_batch(all_image_paths)
        
        # 计算相似度
        similarities = []
        for img_path in all_image_paths:
            if img_path in all_features and img_path != query_image.replace("./Pic/", ""):
                sim = self.calculate_similarity(query_feat, all_features[img_path])
                similarities.append((img_path, sim))
        
        # 按相似度排序并返回top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_merge_suggestions_with_images(self, group_id):
        """获取合并建议以及相关组的图片"""
        merge_suggestions_with_images = []
        for suggestion in self.verification_report["merge_suggestions"]:
            if suggestion["group_a"] == group_id or suggestion["group_b"] == group_id:
                # 获取另一个组的ID
                other_group_id = suggestion["group_b"] if suggestion["group_a"] == group_id else suggestion["group_a"]
                
                # 获取两个组的图片
                group_a_images = self.get_group_images(suggestion["group_a"])
                group_b_images = self.get_group_images(suggestion["group_b"])
                
                merge_suggestions_with_images.append({
                    'suggestion': suggestion,
                    'other_group_id': other_group_id,
                    'group_a_images': group_a_images,
                    'group_b_images': group_b_images
                })
        
        return merge_suggestions_with_images
    
    def get_low_similarity_images(self, group_id):
        """获取组中低相似度的图片"""
        for entry in self.verification_report["confirmed_groups"]:
            if entry['product_id'] == group_id:
                return entry.get('low_similarity_images', [])
        return []
    
    def calculate_candidate_similarity(self, group_id, candidate_images):
        """计算候选图片与当前组的相似度"""
        if not self.feature_extractor:
            # 如果没有特征提取器，返回空相似度
            return {img: 0.0 for img in candidate_images}
        
        # 获取当前组的query图片
        group_query_images = set()
        for entry in self.similarity_data:
            if entry['product_id'] == group_id:
                group_query_images.add(entry['query_image'])
        
        # 如果找不到组的query图片，使用组内的第一张图片作为代表
        if not group_query_images:
            group_images = self.get_group_images(group_id)
            if group_images:
                group_query_images.add(group_images[0])
            else:
                return {img: 0.0 for img in candidate_images}
        
        # 提取所有需要计算的图片特征
        images_to_process = list(set(list(group_query_images) + candidate_images))
        features = self.feature_extractor.extract_features_batch(images_to_process)
        
        # 计算每个候选图片与组查询图片的平均相似度
        similarities = {}
        for cand_img in candidate_images:
            if cand_img in features:
                # 计算候选图片与组中所有query图片的平均相似度
                max_similarity = 0.0
                for query_img in group_query_images:
                    if query_img in features:
                        similarity = self.calculate_similarity(features[cand_img], features[query_img])
                        max_similarity = max(max_similarity, similarity)
                similarities[cand_img] = max_similarity
            else:
                similarities[cand_img] = 0.0
        
        return similarities
    
    def calculate_similarity(self, feat1, feat2):
        """计算余弦相似度"""
        if feat1 is None or feat2 is None:
            return 0.0
        # 计算余弦相似度
        similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
        return max(0.0, similarity)

    def undo_last_decision(self):
        """撤销上一次决策"""
        if not self.review_state.get("undo_stack"):
            return {"success": False, "error": "没有可撤销的操作"}
        
        # 弹出最后一个决策
        last_undo_info = self.review_state["undo_stack"].pop()
        
        group_id = last_undo_info["group_id"]
        candidate_image = last_undo_info["candidate_image"]
        decision = last_undo_info["decision"]
        previous_state = last_undo_info["previous_state"]
        
        print(f"撤销操作 - 组ID: {group_id}, 候选图片: {candidate_image}, 决策: {decision}")
        
        try:
            # 恢复到之前的状态
            for entry in self.similarity_data:
                if entry['product_id'] == group_id and entry['query_image'] in previous_state:
                    entry['relevant_images'] = previous_state[entry['query_image']]['relevant_images'][:]
            
            # 也更新内部组数据结构
            for category, group_data in self.groups_by_product[group_id].items():
                # 重新构建图片集合，基于恢复后的数据
                group_data['images'] = set()
                for entry in self.similarity_data:
                    if entry['product_id'] == group_id:
                        group_data['images'].add(entry['query_image'])
                        for rel_img in entry['relevant_images']:
                            group_data['images'].add(rel_img)
            
            # 从group_decisions中移除最后一条记录
            if group_id in self.review_state["group_decisions"]:
                if self.review_state["group_decisions"][group_id]:
                    self.review_state["group_decisions"][group_id].pop()
                    # 如果决策列表为空，删除整个组的决策记录
                    if not self.review_state["group_decisions"][group_id]:
                        del self.review_state["group_decisions"][group_id]
            
            # 保存恢复后的数据
            self.save_similarity_data()
            self.save_review_state()
            
            print(f"✓ 成功撤销决策 - 组 {group_id}, 图片 {candidate_image}")
            return {
                "success": True, 
                "message": f"已撤销对图片 {candidate_image} 的决策",
                "group_id": group_id,
                "candidate_image": candidate_image
            }
        except Exception as e:
            print(f"撤销操作失败: {str(e)}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            # 将撤销信息放回栈中，因为操作失败了
            self.review_state["undo_stack"].append(last_undo_info)
            return {"success": False, "error": str(e)}

    def get_review_statistics(self):
        """获取审核统计"""
        total_groups = len(self.get_all_groups())
        
        # 计算已完成的决策数
        total_completed_decisions = 0
        for group_id, decisions in self.review_state["group_decisions"].items():
            total_completed_decisions += len(decisions)
        
        # 计算已处理的组数（基于是否有决策记录）
        processed_groups = 0
        for group_id, decisions in self.review_state["group_decisions"].items():
            if len(decisions) > 0:  # 如果该组有至少一个决策，则认为该组已开始处理
                processed_groups += 1
        
        # 更新已处理组列表
        self.review_state["processed_groups"] = list(self.review_state["group_decisions"].keys())
        
        # 假设每个组平均需要约10个决策来充分审核（这是一个估算值）
        # 或者，我们可以设定一个更合理的最大决策数，例如每个组最多10个决策
        total_possible_decisions = total_groups * 10  # 调整为更合理的数值
        
        return {
            "total": total_groups,
            "processed": processed_groups,
            "remaining": total_groups - processed_groups,
            "total_decisions": total_completed_decisions,
            "total_possible_decisions": total_possible_decisions,
            "progress": (processed_groups / total_groups * 100) if total_groups > 0 else 0,
            "decision_progress": min((total_completed_decisions / total_possible_decisions * 100), 100) if total_possible_decisions > 0 else 0
        }

    def add_to_undo_stack(self, undo_info):
        """添加撤销信息到栈"""
        self.review_state["undo_stack"].append(undo_info)
        self.save_review_state()

    def mark_candidates_as_processed(self, group_id, candidate_images):
        """标记候选图片为已处理"""
        # 此方法不再记录为决策，因为决策已在其他地方记录
        # 只需保存状态即可
        self.save_review_state()


# 全局变量
manager = None


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    global manager
    
    # 假设相似图片数据在当前目录
    similarity_data_path = "./similarity_annotations.json"
    
    if os.path.exists(similarity_data_path):
        manager = GroupReviewManager(
            similarity_data_path=similarity_data_path,
            base_path="./Pic"
        )
        print(f"✓ 按组审核管理器已加载: {similarity_data_path}")
        
        stats = manager.get_review_statistics()
        print(f"  总组数: {stats['total']}")
        print(f"  已处理: {stats['processed']}")
        print(f"  待处理: {stats['remaining']}")
    else:
        print(f"⚠ 未找到相似图片数据文件: {similarity_data_path}")


@app.get("/")
async def root_redirect():
    """根路径重定向到主页"""
    return RedirectResponse(url="/group-home", status_code=302)


@app.get("/group-home", response_class=HTMLResponse)
async def group_home(request: Request):
    """按组审核主页"""
    if not manager:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": "未找到相似图片数据文件，请确保已生成similarity_annotations.json"
        })
    
    stats = manager.get_review_statistics()
    groups = manager.get_all_groups()
    
    return templates.TemplateResponse("group_home.html", {
        "request": request,
        "stats": stats,
        "total_groups": len(groups),
        "categories": ['buildingBlock', 'watergun']
    })


@app.get("/group-review", response_class=HTMLResponse)
async def group_review(request: Request, review_type: str = None, target_group: str = None):
    """获取待审核的图片组，根据验证状态展示不同界面"""
    try:
        manager.load_review_state()
        
        # 如果指定了目标组，直接加载该组
        if target_group:
            group_id = target_group
            # 获取当前组的信息
            group_entries = [entry for entry in manager.similarity_data if entry['product_id'] == group_id]
            if not group_entries:
                return templates.TemplateResponse("error.html", {
                    "request": request,
                    "message": f"未找到组: {group_id}"
                })
            category = group_entries[0]['category'] if group_entries else "unknown"
        else:
            # 否则获取下一个待审核的组
            result = manager.get_next_candidate_for_group()
            if not result:
                stats = manager.get_review_statistics()
                return templates.TemplateResponse("review_complete.html", {
                    "request": request,
                    "stats": stats,
                    "message": "所有图片组已完成审核！"
                })
            group_id, candidate_image, category = result
        
        # 获取当前组的所有图片
        group_images = manager.get_group_images(group_id)
        
        # 分离query_images和relevant_images
        query_images = []
        relevant_images = []
        
        # 遍历相似度数据找到该组的所有图片
        for entry in manager.similarity_data:
            if entry['product_id'] == group_id:
                query_images.append(entry['query_image'])
                relevant_images.extend(entry['relevant_images'])
        
        # 去重relevant_images
        relevant_images = list(set(relevant_images))
        
        # 获取组的验证状态
        verification_status, avg_similarity = manager.get_group_verification_status(group_id)
        
        # 根据验证状态决定展示的内容
        top_similar_images = []
        low_similarity_images = []
        merge_suggestions_with_images = []
        
        if verification_status == 'confirmed_high_sim' or review_type == 'verified':
            # 1. verified_annotations情况：展示与query_image最相似的top20图片
            if query_images:
                top_similar_images = manager.get_top_similar_images(query_images[0], top_k=20)
        
        elif verification_status == 'needs_split' or review_type == 'split':
            # 2. split_suggestions情况：展示组内低相似度图片
            low_similarity_images = manager.get_low_similarity_images(group_id)
        
        elif verification_status == 'has_low_similarity_images' or review_type == 'low_sim':
            # 也是低相似度图片情况
            low_similarity_images = manager.get_low_similarity_images(group_id)
        
        elif review_type == 'merge':
            # 3. merge_suggestions情况：展示合并建议
            merge_suggestions_with_images = manager.get_merge_suggestions_with_images(group_id)
        
        # 获取所有类型的建议
        merge_suggestions = manager.get_merge_suggestions_for_group(group_id)
        split_suggestions = manager.get_split_suggestions_for_group(group_id)
        
        # 计算进度
        total_groups = len(manager.groups_by_product)
        processed_groups = sum(1 for decisions in manager.review_state.get("group_decisions", {}).values() if decisions)
        progress_percentage = (processed_groups / total_groups) * 100 if total_groups > 0 else 0
        
        stats = manager.get_review_statistics()
        return templates.TemplateResponse("group_review.html", {
            "request": request,
            "group_id": group_id,
            "category": category,
            "group_images": group_images,
            "current_progress": get_current_progress(),
            "total_groups": total_groups,
            "processed_groups": processed_groups,
            "progress_percentage": progress_percentage,
            "query_images": query_images,
            "relevant_images": relevant_images,
            "verification_status": verification_status,
            "avg_similarity": avg_similarity,
            "merge_suggestions": merge_suggestions,
            "split_suggestions": split_suggestions,
            "low_similarity_images": low_similarity_images,
            "top_similar_images": top_similar_images,
            "merge_suggestions_with_images": merge_suggestions_with_images,
            "review_type": review_type,
            "stats": stats,
            "group_data": {
                "group_id": group_id,
                "category": category,
                "query_images": query_images,
                "relevant_images": relevant_images,
                "total_images": len(query_images) + len(relevant_images)
            }
        })
    except Exception as e:
        print(f"获取图片组审核页面失败: {str(e)}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/group-decision")
async def process_group_decision(
    group_id: str = Form(...),
    candidate_image: str = Form(...),
    decision: str = Form(...)
):
    """处理单个组决策"""
    if not manager:
        return JSONResponse({"success": False, "error": "审核管理器未初始化"})
    
    print(f"接收到的表单数据 - 组ID: '{group_id}', 候选图片: '{candidate_image}', 决策: '{decision}'")
    
    # 检查决策值是否为空
    if not decision or decision == "null":
        print(f"警告: 接收到无效的决策值: '{decision}'")
        return JSONResponse({"success": False, "error": "决策值不能为空"})
    
    try:
        print(f"处理决策 - 组ID: {group_id}, 候选图片: {candidate_image}, 决策: {decision}")
        manager.process_group_decision(group_id, candidate_image, decision)
        print(f"决策处理完成 - 组ID: {group_id}")
        return RedirectResponse(url="/group-review", status_code=303)
    except Exception as e:
        print(f"处理错误: {str(e)}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return JSONResponse({"success": False, "error": str(e)})


@app.post("/save-final-result")
async def save_final_result():
    """保存最终结果到similarity_annotations_final.json"""
    if not manager:
        return JSONResponse({"success": False, "error": "审核管理器未初始化"})
    
    try:
        # 保存最终结果到单独的文件
        final_path = manager.similarity_data_path.replace('.json', '_final.json')
        with open(final_path, 'w', encoding='utf-8') as f:
            json.dump(manager.similarity_data, f, indent=2, ensure_ascii=False)
        print(f"✓ 已手动保存最终结果到: {final_path}")
        
        return JSONResponse({
            "success": True,
            "message": f"最终结果已保存到 {final_path}",
            "final_path": final_path
        })
    except Exception as e:
        print(f"保存最终结果时发生错误: {str(e)}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return JSONResponse({"success": False, "error": str(e)})


@app.post("/batch-group-decision")
async def batch_group_decision(request: Request):
    """处理批量组决策"""
    try:
        form_data = await request.form()
        group_id = form_data.get('group_id')
        
        # 获取URL参数中的review_type
        url_params = dict(request.query_params)
        review_type = url_params.get('review_type', '')
        
        # 根据不同模式获取选中的图片
        if review_type == 'verified':
            # 在verified模式下，使用included_images作为参数名
            selected_images = form_data.getlist('included_images')
        else:
            # 在其他模式下，使用candidates作为参数名
            selected_images = form_data.getlist('candidates')
        
        print(f"接收到批量决策请求:")
        print(f"  组ID: {group_id}")
        print(f"  审核类型: {review_type}")
        print(f"  选中的图片: {selected_images}")
        
        if not group_id:
            return RedirectResponse(url=f"/group-review?review_type={review_type}", status_code=303)
        
        # 获取当前组的所有候选图片
        all_candidates = manager.get_group_candidates(group_id) if review_type != 'verified' else []
        
        # 确定被选中和被排除的图片
        included_images = set(selected_images)  # 被勾选的图片（包含）
        excluded_images = [img for img in all_candidates if img not in included_images] if all_candidates else []
        
        print(f"  包含的图片: {included_images}")
        print(f"  排除的图片: {excluded_images}")
        
        # 在处理前记录原始状态用于撤销
        original_states = {}
        for entry in manager.similarity_data:
            if entry['product_id'] == group_id:
                original_states[entry['query_image']] = {"relevant_images": entry['relevant_images'][:]}

        # 更新相似度数据 - 对该组中的每一个条目都更新
        for entry in manager.similarity_data:
            if entry['product_id'] == group_id:
                if review_type == 'verified':
                    # 对于verified类型，直接将选中的图片设为relevant_images
                    # 确保query_image也在relevant_images中
                    new_relevant_images = list(included_images)
                    if entry['query_image'] not in new_relevant_images:
                        new_relevant_images.insert(0, entry['query_image'])
                    entry['relevant_images'] = new_relevant_images
                else:
                    # 对于其他类型，按照原有逻辑处理包含和排除的图片
                    # 处理包含的图片
                    for img in included_images:
                        if img != entry['query_image']:  # 不将自身的图片添加到相关图片中
                            if img not in entry['relevant_images']:
                                entry['relevant_images'].append(img)
                    
                    # 处理排除的图片
                    for img in excluded_images:
                        if img in entry['relevant_images']:
                            entry['relevant_images'].remove(img)
                    
                    # 确保relevant_images不包含重复项
                    entry['relevant_images'] = list(set(entry['relevant_images']))

        # 保存更新后的数据
        manager.save_similarity_data()
        
        # 记录决策历史用于撤销（记录所有处理的图片）
        for img in included_images:
            # 记录包含决策
            decision_record = {
                "group_id": group_id,
                "candidate_image": img,
                "decision": "include",
                "timestamp": str(Path(manager.similarity_data_path).stat().st_mtime)
            }
            if group_id not in manager.review_state["group_decisions"]:
                manager.review_state["group_decisions"][group_id] = []
            manager.review_state["group_decisions"][group_id].append(decision_record)
            
            # 添加到撤销栈
            manager.add_to_undo_stack({
                "group_id": group_id,
                "candidate_image": img,
                "decision": "include",
                "previous_state": original_states.copy()
            })
        
        for img in excluded_images:
            # 记录排除决策
            decision_record = {
                "group_id": group_id,
                "candidate_image": img,
                "decision": "exclude",
                "timestamp": str(Path(manager.similarity_data_path).stat().st_mtime)
            }
            if group_id not in manager.review_state["group_decisions"]:
                manager.review_state["group_decisions"][group_id] = []
            manager.review_state["group_decisions"][group_id].append(decision_record)
        
        # 保存状态
        manager.save_review_state()
        
        print("批量决策处理完成")
        # 返回重定向响应而不是JSON，保留review_type参数
        redirect_url = f"/group-review?review_type={review_type}" if review_type else "/group-review"
        return RedirectResponse(url=redirect_url, status_code=303)
        
    except Exception as e:
        print(f"批量决策处理失败: {str(e)}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        redirect_url = f"/group-review?review_type={review_type}" if 'review_type' in locals() else "/group-review"
        return RedirectResponse(url=redirect_url, status_code=303)


@app.post("/undo-decision")
async def undo_decision():
    """撤销上一次决策"""
    if not manager:
        return JSONResponse({"success": False, "error": "审核管理器未初始化"})
    
    try:
        result = manager.undo_last_decision()
        return JSONResponse(result)
    except Exception as e:
        print(f"撤销操作错误: {str(e)}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return JSONResponse({"success": False, "error": str(e)})


@app.get("/api/group-stats")
async def get_final_results(request: Request):
    """获取最终审核结果展示页面"""
    if not manager:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": "审核管理器未初始化"
        })
    
    # 获取完整的相似度数据
    similarity_data = manager.similarity_data
    
    # 整理图片数据用于展示
    grouped_images = []
    for entry in similarity_data:
        group_data = {
            'product_id': entry['product_id'],
            'category': entry['category'],
            'query_images': [entry['query_image']],
            'relevant_images': entry['relevant_images']
        }
        grouped_images.append(group_data)
    
    return templates.TemplateResponse("final_results.html", {
        "request": request,
        "json_data": similarity_data,
        "grouped_images": grouped_images
    })


@app.post("/api/merge-groups")
async def merge_groups(request: Request):
    """合并选中的组"""
    if not manager:
        return JSONResponse({"error": "审核管理器未初始化"}, status_code=500)
    
    try:
        data = await request.json()
        groups_to_merge = data.get('groups', [])
        
        if len(groups_to_merge) < 2:
            return JSONResponse({"error": "至少需要选择2个组进行合并"}, status_code=400)
        
        # 合并逻辑：将后面组的图片合并到第一个组
        target_group = groups_to_merge[0]
        
        for source_group in groups_to_merge[1:]:
            # 找到源组的所有条目
            source_entries = [entry for entry in manager.similarity_data if entry['product_id'] == source_group]
            
            # 将源组的图片添加到目标组
            for source_entry in source_entries:
                target_entries = [entry for entry in manager.similarity_data if entry['product_id'] == target_group and entry['category'] == source_entry['category']]
                
                if target_entries:
                    # 合并相关图片
                    for rel_img in source_entry['relevant_images']:
                        if rel_img not in target_entries[0]['relevant_images']:
                            target_entries[0]['relevant_images'].append(rel_img)
                    
                    # 添加源组的query图片到目标组的相关图片中
                    if source_entry['query_image'] not in target_entries[0]['relevant_images']:
                        target_entries[0]['relevant_images'].append(source_entry['query_image'])
                else:
                    # 如果目标组没有对应类别的条目，创建一个新的
                    new_entry = {
                        'product_id': target_group,
                        'category': source_entry['category'],
                        'query_image': source_entry['query_image'],
                        'relevant_images': source_entry['relevant_images'][:]
                    }
                    manager.similarity_data.append(new_entry)
            
            # 删除源组
            manager.similarity_data = [entry for entry in manager.similarity_data if entry['product_id'] != source_group]
        
        # 保存更改
        manager.save_similarity_data()
        
        return JSONResponse({"message": f"成功合并 {len(groups_to_merge)} 个组到 {target_group}"})
    except Exception as e:
        print(f"合并组失败: {str(e)}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/delete-groups")
async def delete_groups(request: Request):
    """删除选中的组"""
    if not manager:
        return JSONResponse({"error": "审核管理器未初始化"}, status_code=500)
    
    try:
        data = await request.json()
        groups_to_delete = data.get('groups', [])
        
        if not groups_to_delete:
            return JSONResponse({"error": "没有选择要删除的组"}, status_code=400)
        
        # 删除选中的组
        manager.similarity_data = [entry for entry in manager.similarity_data if entry['product_id'] not in groups_to_delete]
        
        # 保存更改
        manager.save_similarity_data()
        
        return JSONResponse({"message": f"成功删除 {len(groups_to_delete)} 个组"})
    except Exception as e:
        print(f"删除组失败: {str(e)}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/delete-images")
async def delete_images(request: Request):
    """删除选中的图片"""
    if not manager:
        return JSONResponse({"error": "审核管理器未初始化"}, status_code=500)
    
    try:
        data = await request.json()
        images_to_delete = data.get('images', [])
        
        if not images_to_delete:
            return JSONResponse({"error": "没有选择要删除的图片"}, status_code=400)
        
        # 按产品ID分组图片
        images_by_product = {}
        for img_info in images_to_delete:
            product_id = img_info['productId']
            image = img_info['image']
            if product_id not in images_by_product:
                images_by_product[product_id] = []
            images_by_product[product_id].append(image)
        
        # 从每个产品的相关图片中删除选中的图片
        for entry in manager.similarity_data:
            product_id = entry['product_id']
            if product_id in images_by_product:
                for image in images_by_product[product_id]:
                    # 从relevant_images中删除图片
                    if image in entry['relevant_images']:
                        entry['relevant_images'].remove(image)
                    # 如果是query_image，则需要特殊处理
                    if entry['query_image'] == image:
                        # 如果query_image被删除，可以选择删除整个条目或替换为其他相关图片
                        # 这里我们选择删除整个条目
                        pass
        
        # 删除包含被删除query_image的条目
        for img_info in images_to_delete:
            product_id = img_info['productId']
            image = img_info['image']
            manager.similarity_data = [
                entry for entry in manager.similarity_data 
                if not (entry['product_id'] == product_id and entry['query_image'] == image)
            ]
        
        # 保存更改
        manager.save_similarity_data()
        
        return JSONResponse({"message": f"成功删除 {len(images_to_delete)} 张图片"})
    except Exception as e:
        print(f"删除图片失败: {str(e)}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/save-changes")
async def save_changes():
    """保存所有更改到JSON文件"""
    if not manager:
        return JSONResponse({"error": "审核管理器未初始化"}, status_code=500)
    
    try:
        manager.save_similarity_data()
        return JSONResponse({"message": "更改已保存成功"})
    except Exception as e:
        print(f"保存更改失败: {str(e)}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/remove-duplicates")
async def remove_duplicates():
    """移除重复的组（相同图片的不同排列）"""
    if not manager:
        return JSONResponse({"error": "审核管理器未初始化"}, status_code=500)
    
    try:
        original_count = len(manager.similarity_data)
        
        # 创建一个集合来存储唯一的图片组合（不考虑query和relevant的顺序）
        seen_combinations = set()
        unique_entries = []
        
        for entry in manager.similarity_data:
            # 创建一个包含所有图片的排序后的元组作为唯一标识
            all_images = [entry['query_image']] + entry['relevant_images']
            all_images_sorted = tuple(sorted(all_images))
            
            # 检查是否已经存在相同的图片组合
            if all_images_sorted not in seen_combinations:
                seen_combinations.add(all_images_sorted)
                unique_entries.append(entry)
            else:
                print(f"发现重复组，已跳过: {entry['product_id']} - {entry['query_image']}")
        
        # 更新相似度数据
        manager.similarity_data = unique_entries
        removed_count = original_count - len(manager.similarity_data)
        
        # 保存更改
        manager.save_similarity_data()
        
        return JSONResponse({"message": f"去重完成，移除了 {removed_count} 个重复组，剩余 {len(manager.similarity_data)} 个唯一组"})
    except Exception as e:
        print(f"去重失败: {str(e)}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/all-images")
async def get_all_images():
    """获取所有图片列表，包括已分组和未分组的"""
    if not manager:
        return JSONResponse({"error": "审核管理器未初始化"}, status_code=500)
    
    try:
        # 获取所有已分组的图片
        grouped_images = set()
        for entry in manager.similarity_data:
            grouped_images.add(entry['query_image'])
            for img in entry['relevant_images']:
                grouped_images.add(img)
        
        # 获取所有图片文件
        import os
        pic_dir = "./Pic"
        all_files = []
        
        for root, dirs, files in os.walk(pic_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    # 获取相对路径
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, ".")
                    # 使用 / 作为路径分隔符以兼容web访问
                    rel_path = rel_path.replace("\\", "/")
                    all_files.append(rel_path)
        
        # 区分已分组和未分组的图片
        ungrouped_images = [img for img in all_files if img not in grouped_images]
        
        return JSONResponse({
            "total_images": len(all_files),
            "grouped_images": len(grouped_images),
            "ungrouped_images": len(ungrouped_images),
            "sample_ungrouped": ungrouped_images[:50]  # 返回前50个未分组图片作为样本
        })
    except Exception as e:
        print(f"获取所有图片失败: {str(e)}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/group-suggestions/{group_id}")
async def get_group_suggestions(group_id: str):
    """获取指定组的建议信息（合并/拆分）"""
    if not manager:
        return JSONResponse({"error": "审核管理器未初始化"}, status_code=500)
    
    try:
        merge_suggestions = manager.get_merge_suggestions_for_group(group_id)
        split_suggestions = manager.get_split_suggestions_for_group(group_id)
        verification_status, avg_similarity = manager.get_group_verification_status(group_id)
        
        return JSONResponse({
            "merge_suggestions": merge_suggestions,
            "split_suggestions": split_suggestions,
            "verification_status": verification_status,
            "avg_similarity": avg_similarity
        })
    except Exception as e:
        print(f"获取组建议失败: {str(e)}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/handle-low-similarity-image")
async def handle_low_similarity_image(request: Request):
    """处理低相似度图片的决策（保留或移除）"""
    if not manager:
        return JSONResponse({"error": "审核管理器未初始化"}, status_code=500)
    
    try:
        data = await request.json()
        group_id = data.get('group_id')
        image_path = data.get('image_path')
        decision = data.get('decision')  # 'keep' 或 'remove'
        
        if not group_id or not image_path or not decision:
            return JSONResponse({"error": "缺少必要参数"}, status_code=400)
        
        # 查找并更新相应的条目
        for entry in manager.similarity_data:
            if entry['product_id'] == group_id:
                if decision == 'remove' and image_path in entry['relevant_images']:
                    # 从relevant_images中移除图片
                    entry['relevant_images'].remove(image_path)
                    print(f"从组 {group_id} 中移除了图片 {image_path}")
                elif decision == 'keep' and image_path not in entry['relevant_images']:
                    # 确保图片在relevant_images中
                    if image_path != entry['query_image']:  # 确保不是query_image
                        if image_path not in entry['relevant_images']:
                            entry['relevant_images'].append(image_path)
                            print(f"将图片 {image_path} 添加到组 {group_id}")
        
        # 保存更改
        manager.save_similarity_data()
        
        return JSONResponse({
            "success": True,
            "message": f"已{ '保留' if decision == 'keep' else '移除' }图片 {image_path}"
        })
    except Exception as e:
        print(f"处理低相似度图片决策失败: {str(e)}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/api/merge-groups")
async def merge_groups(request: Request):
    """合并两个组"""
    if not manager:
        return JSONResponse({"error": "审核管理器未初始化"}, status_code=500)
    
    try:
        data = await request.json()
        group_a = data.get('group_a')
        group_b = data.get('group_b')
        
        if not group_a or not group_b:
            return JSONResponse({"error": "缺少必要参数"}, status_code=400)
        
        print(f"开始合并组 {group_a} 和 {group_b}")
        
        # 获取两个组的索引
        group_a_index = -1
        group_b_index = -1
        
        for i, entry in enumerate(manager.similarity_data):
            if entry['product_id'] == group_a:
                group_a_index = i
            elif entry['product_id'] == group_b:
                group_b_index = i
        
        if group_a_index == -1 or group_b_index == -1:
            return JSONResponse({"error": "找不到指定的组"}, status_code=404)
        
        # 合并两个组
        group_a_entry = manager.similarity_data[group_a_index]
        group_b_entry = manager.similarity_data[group_b_index]
        
        # 合并relevant_images
        merged_relevant_images = list(set(group_a_entry['relevant_images'] + group_b_entry['relevant_images']))
        
        # 更新第一个组的图片集合
        group_a_entry['relevant_images'] = merged_relevant_images
        
        # 从数据中移除第二个组
        manager.similarity_data.pop(group_b_index)
        
        print(f"成功合并组 {group_a} 和 {group_b}")
        
        # 保存更改
        manager.save_similarity_data()
        
        return JSONResponse({
            "success": True,
            "message": f"成功合并组 {group_a} 和 {group_b}"
        })
    except Exception as e:
        print(f"合并组失败: {str(e)}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/ungrouped-images")
async def get_ungrouped_images(page: int = 1, size: int = 50):
    """获取未分组的图片列表（支持分页）"""
    if not manager:
        return JSONResponse({"error": "审核管理器未初始化"}, status_code=500)
    
    try:
        # 获取所有已分组的图片
        grouped_images = set()
        for entry in manager.similarity_data:
            grouped_images.add(entry['query_image'])
            for img in entry['relevant_images']:
                grouped_images.add(img)
        
        # 获取所有图片文件
        import os
        pic_dir = "./Pic"
        all_files = []
        
        for root, dirs, files in os.walk(pic_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    # 获取相对路径
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, ".")
                    # 使用 / 作为路径分隔符以兼容web访问
                    rel_path = rel_path.replace("\\", "/")
                    all_files.append(rel_path)
        
        # 区分已分组和未分组的图片
        ungrouped_images = [img for img in all_files if img not in grouped_images]
        
        # 分页处理
        total_count = len(ungrouped_images)
        total_pages = (total_count + size - 1) // size  # 向上取整
        page = max(1, min(page, total_pages))  # 确保页码在有效范围内
        
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        page_images = ungrouped_images[start_idx:end_idx]
        
        return JSONResponse({
            "images": page_images,
            "total_count": total_count,
            "total_pages": total_pages,
            "current_page": page,
            "page_size": len(page_images)
        })
    except Exception as e:
        print(f"获取未分组图片失败: {str(e)}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/api/view-ungrouped")
async def view_ungrouped(request: Request):
    """显示未分组图片页面"""
    return templates.TemplateResponse("ungrouped_images.html", {"request": request})


@app.post("/api/create-group-from-images")
async def create_group_from_images(request: Request):
    """从选中的图片创建新组"""
    if not manager:
        return JSONResponse({"error": "审核管理器未初始化"}, status_code=500)
    
    try:
        data = await request.json()
        images = data.get('images', [])
        
        if not images:
            return JSONResponse({"error": "没有选择图片"}, status_code=400)
        
        # 创建新组 - 使用第一个图片作为query_image，其余作为relevant_images
        query_image = images[0]
        relevant_images = images[1:] if len(images) > 1 else []
        
        # 生成新的产品ID（使用时间戳+随机数确保唯一性）
        import time
        import random
        new_product_id = f"custom_group_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # 默认类别，可以根据文件路径推断
        category = "unknown"
        if '/' in query_image:
            category = query_image.split('/')[1] if len(query_image.split('/')) > 1 else "unknown"
        
        new_entry = {
            'product_id': new_product_id,
            'category': category,
            'query_image': query_image,
            'relevant_images': relevant_images
        }
        
        # 添加到相似度数据
        manager.similarity_data.append(new_entry)
        
        # 保存更改
        manager.save_similarity_data()
        
        return JSONResponse({
            "message": f"成功创建新组 '{new_product_id}'，包含 {len(images)} 张图片"
        })
    except Exception as e:
        print(f"创建组失败: {str(e)}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return JSONResponse({"error": str(e)}, status_code=500)


def get_current_progress():
    """获取当前进度"""
    if not manager:
        return {"current": 0, "total": 0}
    
    total_groups = len(manager.groups_by_product)
    processed_groups = len(manager.review_state.get("processed_groups", []))
    return {"current": processed_groups, "total": total_groups}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006, reload=False)
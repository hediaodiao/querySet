#!/usr/bin/env python3
"""
人工审核工具 - 可视化界面
用于审核合并建议和分组结果
"""

import os
import json
import shutil
import re
from pathlib import Path
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn


app = FastAPI(title="图片分组合并与审核工具")

# 挂载静态文件目录（用于显示图片）
app.mount("/static", StaticFiles(directory="./picTest"), name="static")
app.mount("/static_data", StaticFiles(directory="."), name="static_data")

# 设置模板目录
templates = Jinja2Templates(directory="templates")


class HumanReviewManager:
    def __init__(self, base_path, review_report_path):
        self.base_path = base_path
        self.review_report_path = review_report_path
        self.load_review_data()
    
    def load_review_data(self):
        """加载审核数据"""
        with open(self.review_report_path, 'r', encoding='utf-8') as f:
            self.review_data = json.load(f)
        
        # 初始化审核状态
        self.review_state = {
            "processed_merge_suggestions": [],
            "final_groups": {},
            "user_decisions": []
        }
        
        # 加载已有的审核状态（如果有）
        state_file = self.review_report_path.replace('_review_report.json', '_review_state.json')
        if os.path.exists(state_file):
            with open(state_file, 'r', encoding='utf-8') as f:
                self.review_state = json.load(f)
    
    def save_review_state(self):
        """保存审核状态"""
        state_file = self.review_report_path.replace('_review_report.json', '_review_state.json')
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(self.review_state, f, indent=2, ensure_ascii=False)
    
    def get_next_merge_suggestion(self):
        """获取下一个待审核的合并建议"""
        processed_ids = set(self.review_state["processed_merge_suggestions"])
        for suggestion in self.review_data["merge_suggestions"]:
            suggestion_id = f"{suggestion['group_a']}_{suggestion['group_b']}"
            if suggestion_id not in processed_ids:
                return suggestion, suggestion_id
        return None, None
    
    def process_merge_decision(self, suggestion_id, decision, group_a, group_b):
        """处理合并决策"""
        decision_record = {
            "suggestion_id": suggestion_id,
            "decision": decision,
            "group_a": group_a,
            "group_b": group_b,
            "timestamp": str(Path(self.review_report_path).stat().st_mtime)
        }
        
        self.review_state["user_decisions"].append(decision_record)
        self.review_state["processed_merge_suggestions"].append(suggestion_id)
        
        # 如果用户选择了合并，则更新最终分组
        if decision == "merge":
            # 这里应该实现合并逻辑，暂时记录决策
            print(f"用户选择合并: {group_a} 和 {group_b}")
        
        self.save_review_state()
    
    def get_merge_statistics(self):
        """获取合并审核统计"""
        total = len(self.review_data["merge_suggestions"])
        processed = len(self.review_state["processed_merge_suggestions"])
        remaining = total - processed
        return {
            "total": total,
            "processed": processed,
            "remaining": remaining,
            "progress": (processed / total * 100) if total > 0 else 0
        }


# 全局变量
review_manager = None


@app.on_event("startup")
async def startup_event():
    """应用启动时初始化"""
    global review_manager
    
    # 假设审核报告在当前目录
    review_report_path = "./similarity_annotations_review_report.json"
    
    if os.path.exists(review_report_path):
        review_manager = HumanReviewManager(base_path="./picTest", review_report_path=review_report_path)
        print(f"✓ 审核管理器已加载: {review_report_path}")
    else:
        print(f"⚠ 未找到审核报告文件: {review_report_path}")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """主页 - 显示审核进度和选项"""
    if not review_manager:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": "未找到审核报告文件，请确保已生成similarity_annotations_review_report.json"
        })
    
    stats = review_manager.get_merge_statistics()
    
    return templates.TemplateResponse("review_home.html", {
        "request": request,
        "stats": stats,
        "summary": review_manager.review_data.get("summary", {})
    })


@app.get("/merge-review", response_class=HTMLResponse)
async def merge_review(request: Request):
    """合并建议审核页面"""
    if not review_manager:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": "未找到审核报告文件"
        })
    
    suggestion, suggestion_id = review_manager.get_next_merge_suggestion()
    
    if not suggestion:
        return templates.TemplateResponse("review_complete.html", {
            "request": request,
            "message": "所有合并建议已完成审核！"
        })
    
    # 获取两个分组的图片
    group_a_images = get_images_for_group(suggestion['group_a'])
    group_b_images = get_images_for_group(suggestion['group_b'])
    
    stats = review_manager.get_merge_statistics()
    
    return templates.TemplateResponse("merge_review.html", {
        "request": request,
        "suggestion": suggestion,
        "suggestion_id": suggestion_id,
        "group_a_images": group_a_images,
        "group_b_images": group_b_images,
        "similarity_score": suggestion['similarity_score'],
        "stats": stats
    })


@app.post("/merge-decision")
async def process_merge_decision(
    suggestion_id: str = Form(...),
    decision: str = Form(...),
    group_a: str = Form(...),
    group_b: str = Form(...)
):
    """处理合并决策"""
    if not review_manager:
        return JSONResponse({"success": False, "error": "审核管理器未初始化"})
    
    try:
        review_manager.process_merge_decision(suggestion_id, decision, group_a, group_b)
        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)})


@app.get("/api/stats")
async def get_stats():
    """获取审核统计API"""
    if not review_manager:
        return {"error": "审核管理器未初始化"}
    
    return review_manager.get_merge_statistics()


@app.get("/api/current-suggestion")
async def get_current_suggestion():
    """获取当前待审核的建议"""
    if not review_manager:
        return {"error": "审核管理器未初始化"}
    
    suggestion, suggestion_id = review_manager.get_next_merge_suggestion()
    if not suggestion:
        return {"has_more": False}
    
    # 获取两个分组的图片
    group_a_images = get_images_for_group(suggestion['group_a'])
    group_b_images = get_images_for_group(suggestion['group_b'])
    
    return {
        "has_more": True,
        "suggestion": suggestion,
        "suggestion_id": suggestion_id,
        "group_a_images": group_a_images,
        "group_b_images": group_b_images
    }


def get_images_for_group(group_key):
    """根据分组键获取图片列表"""
    # 这里需要解析分组键并获取对应的图片
    # group_key 格式可能是类似 "12345678_cluster_0" 的形式
    images = []
    
    # 从所有图片中查找匹配该分组的图片
    if os.path.exists("./picTest"):
        for category in ['buildingBlock', 'watergun']:
            category_path = os.path.join("./picTest", category)
            if os.path.exists(category_path):
                for file in os.listdir(category_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        # 检查文件名是否包含分组键的前缀（前8位数字）
                        file_digits_match = re.search(r'(\d+)', file)
                        if file_digits_match:
                            file_digits = file_digits_match.group(1)
                            # 检查文件名中的数字前缀是否与分组键匹配
                            if file_digits.startswith(group_key.split('_')[0]):
                                images.append(f"{category}/{file}")
    
    # 如果上面的方法没找到图片，就从所有图片中随机选择一些
    if not images and os.path.exists("./picTest"):
        for root, dirs, files in os.walk("./picTest"):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, "./picTest")
                    images.append(rel_path)
                    if len(images) >= 10:  # 限制显示数量
                        break
            if len(images) >= 10:
                break
    
    return images[:6]  # 限制显示6张图片


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)
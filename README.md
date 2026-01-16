# 图片分组三阶段流水线

本项目实现了完整的 "文件名规则初筛 → 模型视觉校验 → 人工最终审核" 流程。

## 📋 流程概述

### 第一阶段：文件名规则初筛 (`main2.py`)
- 根据文件名前8位数字对图片进行初步分组
- 生成 `similarity_annotations.json` 文件

### 第二阶段：模型视觉校验 (`model_verification.py`) 
- 使用颜色直方图方法计算图片视觉相似度
- 确认高相似度分组
- 建议合并跨ID相似分组
- 建议拆分低相似度分组
- 生成 `similarity_annotations_verification_report.json`

### 第三阶段：人工最终审核 (`group_review_tool.py`)
- 提供Web界面进行人工审核
- 可以查看、修改、确认分组结果
- 保留初始备份和最终结果

## 📁 文件说明

- `similarity_annotations.json` - 当前活动的工作文件
- `similarity_annotations_initial.json` - 初始分组备份（由main2.py生成）
- `similarity_annotations_verification_report.json` - 模型校验报告
- `similarity_annotations_group_review_state.json` - 审核进度状态

## 🚀 使用方法

### 完整流程：
```bash
# 1. 运行流水线控制器（推荐）
python pipeline_controller.py

# 或者分步执行：
# 第一步：文件名规则分组
python main2.py

# 第二步：模型视觉校验  
python model_verification.py

# 第三步：人工审核
python group_review_tool.py
```

### 访问审核界面：
- 启动 `group_review_tool.py` 后访问 `http://localhost:8004`
- 可以查看分组结果、进行编辑、删除等操作

## 🔧 特性

- **安全性**：所有修改都会保留初始备份
- **智能性**：模型自动识别高相似度组合并建议
- **灵活性**：人工审核可纠正模型错误
- **可视化**：Web界面便于操作和查看

## 📊 校验结果说明

- **确认分组**：模型确认视觉相似度高的组（无需人工审核）
- **建议合并**：模型建议合并的跨ID组（需人工确认）  
- **建议拆分**：模型识别出的低相似度组（需人工处理）
- **需审核**：中等相似度的组（需人工判断）

通过这个三阶段流水线，实现了效率与准确性的最佳平衡。
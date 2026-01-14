import json
from PIL import Image
import os

class AnnotationReviewer:
    """人工审核工具类"""
    
    def __init__(self, annotation_file, image_base_path):
        self.annotation_file = annotation_file
        self.image_base_path = image_base_path
        with open(annotation_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
    
    def show_image_group(self, product_id, category):
        """显示指定产品ID的所有图片"""
        group_images = []
        for entry in self.data:
            if entry['product_id'] == product_id and entry['category'] == category:
                group_images.append(entry['query_image'])
                group_images.extend(entry['relevant_images'])
        
        # 去重
        unique_images = list(set(group_images))
        print(f"产品 {product_id} 共有 {len(unique_images)} 张图片:")
        
        # 这里可以添加显示图片的代码（如在Jupyter Notebook中）
        # 或者生成一个HTML页面供审核
        return unique_images
    
    def generate_review_html(self, output_html="review_report.html"):
        """生成便于审核的HTML报告"""
        html_content = """
        <html>
        <head>
            <title>相似图片标注审核报告</title>
            <style>
                .product-group { margin: 20px; padding: 10px; border: 1px solid #ccc; }
                .images-container { display: flex; flex-wrap: wrap; }
                .image-item { margin: 5px; }
                img { max-width: 200px; max-height: 200px; }
            </style>
        </head>
        <body>
            <h1>相似图片标注审核报告</h1>
        """
        
        # 按产品ID分组
        products = defaultdict(list)
        for entry in self.data:
            key = (entry['product_id'], entry['category'])
            products[key].append(entry)
        
        for (product_id, category), entries in products.items():
            html_content += f'<div class="product-group">'
            html_content += f'<h2>产品ID: {product_id} | 类别: {category}</h2>'
            html_content += f'<p>图片数量: {len(entries)} 个查询对</p>'
            html_content += '<div class="images-container">'
            
            # 获取所有图片路径
            all_images = set()
            for entry in entries:
                all_images.add(entry['query_image'])
                all_images.update(entry['relevant_images'])
            
            for img_path in all_images:
                full_path = os.path.join(self.image_base_path, img_path)
                if os.path.exists(full_path):
                    html_content += f'<div class="image-item">'
                    html_content += f'<img src="{full_path}" alt="{img_path}">'
                    html_content += f'<br><span>{os.path.basename(img_path)}</span>'
                    html_content += '</div>'
            
            html_content += '</div></div>'
        
        html_content += '</body></html>'
        
        with open(output_html, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"审核报告已生成: {output_html}")

# 使用示例
if __name__ == "__main__":
    reviewer = AnnotationReviewer("similarity_annotations.json", r"D:\Project\dateget3")
    reviewer.generate_review_html()
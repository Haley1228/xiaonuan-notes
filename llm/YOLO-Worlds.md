---
data: 2026-04-22
---
# 基本流程

1. 载入一个带ClipPriorDetect的YOLO-Master模型
2. 读取Neu.yaml 数据集配置
3. 从names里拿类别名
4. 用CLIP把类别名编码成文本特征
5. 图像走YOLO-Master主干提特征
6. 检测头同时算：
    - 框回归
    - 原始分类分数
    - CLIP 文本先验分数
7. 分类分数相加
8. 
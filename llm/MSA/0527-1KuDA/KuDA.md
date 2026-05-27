---
data: 2026-05-27
tags:
---
# 1.背景

1. 先前的方法均等对待每个模态的贡献或将文本作为主要模态，忽略了每种模态成为主导的可能
# 2.相关工作

1. 三元对称（两两模态双向关系建模）与基于文本为中心的方法[[KuDA#*亮点]]
2. 对比学习
# \*亮点

1. Introduction中的关于三元对称法与以文本主导方法的概述
> when the dominant modality is not fixed, the ternary
 symmetric-based methods cannot effectively adapt
 to the situation where any modality is dominant
 because they do not consider the differences of im-
 portance between modalities. The text center-based
 methods statically set text as the dominant modal-
 ity, and when other modalities are dominant, the
 model’s attention is distracted by the text.

因此，当主导模态不固定时，基于三元对称的方法无法有效地适应任何模态占主导地位的情况，因为分不清各个模态重要性的区别。基于文本中心的方法静态地将文本设为主导模态，当其他模态占优时，模型的注意力会被文本分散。
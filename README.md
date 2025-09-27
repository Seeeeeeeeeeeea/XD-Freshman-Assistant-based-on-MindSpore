## 项目概述

**西电新生百事通** 是一个基于昇思MindSpore框架和Qwen2.5-7B模型开发的人工智能校园政策问答系统，旨在解决西安电子科技大学（西电）师生及校外人员在获取校园政策信息时面临的“信息碎片化、交互不友好、时效性差”等问题。通过对校园政策文档的微调和优化，项目实现了高效、精准的政策查询服务，助力新生快速融入校园，提升校园服务效率。

---

**团队名称**: Model Evolvers  
**成员**:

- 杨佳豪（西安电子科技大学，大数据管理与应用，2023级）：负责数据合成和模型训练
- 王睿奇（西安电子科技大学，大数据管理与应用，2023级）：负责微调模型训练
- 杨牧野（西安电子科技大学，大数据管理与应用，2023级）：负责数据收集与演示文稿制作

**团队理念**: 以“进化者”为名，致力于通过不断学习、迭代与优化，推动人工智能模型性能提升，赋能校园服务场景。

---
代码结构
```
your_project_path
|   formatted.json
|   inference.py
|   README.md
|   requirement.txt
|   train.py
|   培训.ipynb
|   推理.ipynb
|
+---yourdata
|
+---output/your_weights/
\---Qwen2.5-7B
```

你可以直接使用实现环境配置
```bash
pip install -r requirements.txt
```
环境配置好之后，你可以通过下面操作下载Qwen2.5-7B模型
```bash
modelscope download --model Qwen/Qwen2.5-7B --local_dir ./Qwen2.5-7B
```
通过运行`train.py来进行训练
```
```bash
python train.py
```
通过运行`inference.py`来实现推理
```bash
python inference.py
```

---

## 参考资料

- [MindSpore官方教程](https://www.mindspore.cn/tutorials/zh-CN/r2.7.0/index.html)
- [Qwen2.5-VL-7B-Instruct模型](https://www.modelscope.cn/models/Qwen/Qwen2.5-VL-7B-Instruct)
- [MindNLP文档](https://mindnlp.cqu.ai/)
- 培训.ipynb
- 合成数据.ipynb
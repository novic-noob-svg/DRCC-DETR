# DRCC-DETR 开源源码说明

仓库：https://github.com/novic-noob-svg/DRCC-DETR

```bash
git clone https://github.com/novic-noob-svg/DRCC-DETR.git
cd DRCC-DETR
```

旧配置中的整体模型注册名称会自动迁移为 `drcc_detr`。此次名称调整不改变模型张量、计算过程和权重文件名。

本目录是独立整理的源码副本，原始项目和训练备份均未覆盖。英文使用说明见 [README.md](README.md)。

## 模块名称

- 总体模型：DRCC-DETR。
- VCR：Visual Contrastive Regularization，视觉对比正则化。
- SADRA：Scale-Adaptive Dual-Region Attention，尺度自适应双区域注意力。
- 原代码中的 VCA 文件、类、运行参数、输出字典及损失项已改为 VCR。历史名称只保留在旧配置/权重迁移层。
- 公开配置使用 `modelname='drcc_detr'`；内部保留 DINO 基础实现及旧 `dino` 构建器，以兼容原训练配置。

## 本次整理基准

以你指定的 `output_aitod/config_args_all.json` 和训练备份为最终 AI-TOD 配置依据。提供了可直接执行的 `config/DRCC_DETR/aitodv2_r50_5scale.py`。

实际设置为 R50、5 个特征尺度、900 queries、24 epochs、每卡 batch size=1、4 卡、EMA 开启、AMP 关闭，VCR 权重 0.5、温度 0.07，SADRA 阈值 0.1、上限 2.5，边界带开启且宽度 0.2。该训练配置没有 `use_ns_cdn`，原构建器的有效默认值为关闭；开源主配置显式设为关闭。

**论文对应性需要区分一处实际差异：最终 AI-TOD 训练启用了边界带投影，而当前论文的 SADRA 主公式只描述尺度扩展与区域先验。此版本保留权重实际使用的行为，没有根据论文叙述删除它，也没有修改论文。**

## 清理内容

已排除数据集、训练权重、输出日志、缓存、旧版本脚本、TinyPerson 专用工具、YOLO 格式转换工具、过期上游新闻/图片、不相关模型项目和 Swin/ConvNeXt 分支。删除了废弃的整段注释实现、q2 系列数据钩子和本机数据复制路径。

保留了 bbox 检测所需的 DINO 主干、CUDA 算子源码、COCO 数据增强/评估、训练与断点恢复、EMA、最佳模型选择、AI-TOD 官方评估、组件消融、SADRA 参数敏感性及 VCR/SADRA 可视化。

额外修正：数据缺失时原加载器会尝试用下一张图替换，现改为直接报错；最终验证严格检查权重完整性；普通训练不再额外保存采样可视化张量。

## 已有权重如何使用

已有权重保持原文件即可，不需要重新训练或手动改参数名称。训练/验证工具会将历史 `vca_feat_proj`、`vca_head` 和 `module.` 前缀迁移到新名称；旧 `config_args_all.json` 的 VCA 字段也会转换。

原权重、数据集没有放入源码压缩包。可以在发布仓库后，通过单独的模型下载位置提供权重，并保留对应完整配置。

## 实验配置边界

AI-TOD 配置有实际训练记录支撑。VisDrone 四尺度文件仅为可运行的配置框架，不声称自动复现全部论文结果。对于 HIT-UAV、VisDrone 主实验和敏感性实验，请用 `tools/export_run_config.py` 导出各自真实的完整配置。

原组件消融清单单独保留：边界带关闭、neighbor-safe CDN 开启；其参考模型与最终 AI-TOD 主结果不是同一训练口径。不能将两组配置混用后归因于单个模块。

安装、训练、评估、恢复和可视化命令均在英文 README 中。开源包已保留上游版权头和 Apache-2.0 许可证；没有替换原作者版权声明。

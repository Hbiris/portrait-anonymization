# Phase 0 实测报告

> 本文保留初始实验记录。官方 RetinaFace 裁剪/对齐对照现已完成，最新结论与逐图指标见 [预处理对照报告](PREPROCESSING_REPORT.md)。

结论：**官方 checkpoint 已在本机运行；当前 baseline 不值得直接进入完整 Phase 1 服务开发。** 先继续定向模型/预处理评估和真实业务数据验收。没有创建 FastAPI、OCR 或服务架构。

## 实际环境与复现范围

- macOS 15.3.2，arm64；MediaPipe 日志识别 Apple M1 Pro。Python 3.8.20（项目隔离环境）。
- torch 2.2.2、torchvision 0.17.2、MediaPipe 0.10.9、OpenCV contrib 4.8.1.78、NumPy 1.24.4、Pillow 10.4.0、Lightning 2.0.9、torchmetrics 1.2.1、SciPy 1.10.1。完整依赖见 [requirements.lock.txt](requirements.lock.txt)。
- 生成器与 OpenCV 检测/识别使用 CPU，torch 4 线程；MediaPipe CPU graph 仍需 macOS GL 上下文，沙箱内创建失败，放到本机隔离环境后成功。没有 CUDA，也未把 MPS 用于本轮生成推理。
- 官方源代码 commit：`7dbe45e3b172670ff59be982c662749165ab5ced`。使用未修改的 GeneratorUNet、ZeroPaddingResize、FacialLandmarks478，严格加载 generator state_dict。
- **不是官方完整 CLI 的原样复现**：为避免研究训练/GUI/TensorFlow 全依赖，使用最小推理适配器；YuNet 替代 RetinaFace 裁剪。对照轮使用 1.0/0.5 尺度检测、NMS、bbox 每侧 20% 上下文、align=False。没有改模型权重或训练。该差异会影响质量，因此不能把所有失败归因于生成器。
- 不做整图回贴：官方生成器输出 512×512 人脸裁剪候选，背景是生成模型的黑底；完整身体/场景保持、整图合成与全图覆盖验证尚未实现。Phase 0 的图片交付是逐脸候选及逐输入失败记录，不是服务可用整图。

## Checkpoint

- 官方 README 的论文版 / 25 epochs 链接：[University of Augsburg](https://mediastore.rz.uni-augsburg.de/get/NsLjQYey65/)。
- 本地：`models/publication-download`；687127579 bytes（约 655.30 MiB）。
- SHA-256：`eba49bd525033b55022a91d6b4398ab088b51339fd6895db7afdd648d776eec9`。这是本地完整下载后的哈希，不是发行方数字签名。
- checkpoint epoch=24（零基，即第 25 个 epoch），global_step=283800；训练 Lightning=2.0.0。超参数 n_epochs=50 是训练计划值，不代表这个权重已训练 50 epochs。
- 生成器 54,404,099 个参数；用 weights_only=True 读取 checkpoint，不加载优化器用于推理。
- 主仓库 MIT 不等于预训练模型商用授权已解决；其 CelebA 训练来源的限制仍待解决。当前仅研究 baseline。

## 数据与评估范围

16 张输入：12 张公开许可照片、4 张派生压力测试。12 张不等于 12 个独立人物：s03/s04 是同一个人的不同照片，s13–s16 与父样本相关。来源、作者、许可和哈希见 [ATTRIBUTION.md](ATTRIBUTION.md) 与 data/sources.json。

| 场景 | 覆盖与缺口 |
|---|---|
| 老年外观人像 | 多张公开照片；没有年龄/护理状态真值，不能凭标题确认年龄 |
| 眼镜、侧脸 | s03/s04、s08、s11 等 |
| 张嘴、进食环境 | s08/s09/s10；不等于已验证吞咽或真实食物遮嘴 |
| 闭眼 | s06 是睡眠候选，眼镜/胡须/伞遮挡；s09 向下看，不能作为经标注的闭眼真值 |
| 低照度 | s13/s16 是亮度乘 0.18 的人工变换，不能替代真实低照度噪声 |
| 遮挡 | 自然手部、眼镜等遮挡；s15 是人工矩形遮挡，不能替代真实食物/器械遮挡 |
| 卧床/侧卧 | **无真实样本**；s14 仅图像旋转 90°，不是侧卧身体/面部视角变化 |

## 实测汇总

- 16 张全部执行并有结果记录；11 张输入产生共 13 张生成脸候选。
- YuNet 共提出 20 个人脸候选（包含小脸、背景图像和潜在误检，不能当成人工确认人数）。
- 12 个生成脸有 SFace identity cosine；12 个有眼嘴/近似姿态指标；11 个同时有两类指标。
- 5 个人脸因输入关键点提取失败未生成；2 个因脸框小于 32 像素拒绝。
- s13/s14/s15 两检测器均返回零脸，但根据已知原图及变换可知仍有人脸，属于检测漏检案例。**多个 detector 均未检出不等于确认没有人脸。** 当前 zero_face_policy=reject；接口支持 continue_if_all_clear，但任何怀疑/错误必须拒绝，该策略也不能绕过已知图像证据。
- 独立于关键点计算的 Codex 视觉复核：13 张候选中 9 张发现明确状态/可见性/表情失真，4 张无法确认。它是可用性反例证据，不是护理专家验收或独立模型验证。
- 当前没有经业务校准的阈值，所有 identity/state pass 均为 null。低相似度不等于匿名化通过，特别是生成脸失真时。0 张获准放行。
- 对照轮样本处理累计约 15.41 秒（不含模型加载、下载、报告生成），含失败快速返回；不是吞吐或 SLA 基准。

## 每张输入/每张脸的指标

N/A 表示未得到可靠结果，绝不按 0 或通过处理。face 编号是检测排序，不是身份。表中 eye A/B 指 MediaPipe 33/263 两侧，不命名解剖学左右眼以避免镜像混淆。

| 输入/脸 | identity cosine | ΔEAR A | ΔEAR B | ΔMAR | 近似姿态差 ° | 执行状态 | 独立 utility 复核 |
|---|---:|---:|---:|---:|---:|---|---|
| s01/f0 | 0.1813 | 0.0175 | 0.0137 | 0.0225 | 3.6778 | candidate_measured | fail |
| s02/f0 | 0.0546 | 0.0094 | 0.0041 | 0.0133 | 5.3248 | candidate_measured | indeterminate |
| s02/f1 | N/A | N/A | N/A | N/A | N/A | source_landmarks_missing | indeterminate |
| s02/f2 | -0.0726 | N/A | N/A | N/A | N/A | validation_indeterminate | fail |
| s03/f0 | 0.0954 | 0.1606 | 0.3730 | 0.2046 | 18.7495 | candidate_measured | fail |
| s04/f0 | N/A | N/A | N/A | N/A | N/A | source_landmarks_missing | indeterminate |
| s05/f0 | 0.1227 | 0.0170 | 0.0037 | 0.0016 | 4.1132 | candidate_measured | indeterminate |
| s06/f0 | 0.1212 | 0.0884 | 0.0621 | 0.1167 | 4.4277 | candidate_measured | fail |
| s07/f0 | N/A | N/A | N/A | N/A | N/A | source_landmarks_missing | indeterminate |
| s08/f0 | 0.0374 | 0.0697 | 0.0580 | 0.1054 | 7.5719 | candidate_measured | fail |
| s08/f1 | N/A | N/A | N/A | N/A | N/A | source_landmarks_missing | indeterminate |
| s09/f0 | 0.0118 | 0.0002 | 0.0938 | 0.0161 | 5.8838 | candidate_measured | indeterminate |
| s10/f0 | N/A | N/A | N/A | N/A | N/A | source_face_too_small | indeterminate |
| s10/f1 | N/A | N/A | N/A | N/A | N/A | source_face_too_small | indeterminate |
| s10/f2 | N/A | 0.0506 | 0.1180 | 0.0011 | 20.8726 | validation_indeterminate | fail |
| s10/f3 | -0.0447 | 0.1179 | 0.1358 | 0.0411 | 3.7112 | candidate_measured | fail |
| s11/f0 | 0.0866 | 0.0209 | 0.1218 | 0.0129 | 5.9586 | candidate_measured | indeterminate |
| s12/f0 | 0.1362 | 0.0240 | 0.0426 | 0.0208 | 4.4020 | candidate_measured | fail |
| s13 | N/A | N/A | N/A | N/A | N/A | reject_zero_face_policy | indeterminate |
| s14 | N/A | N/A | N/A | N/A | N/A | reject_zero_face_policy | indeterminate |
| s15 | N/A | N/A | N/A | N/A | N/A | reject_zero_face_policy | indeterminate |
| s16/f0 | 0.0458 | 0.0559 | 0.0688 | 0.0306 | 3.3817 | candidate_measured | fail |
| s16/f1 | N/A | N/A | N/A | N/A | N/A | source_landmarks_missing | indeterminate |

[浏览全部 16 张输入与候选/失败结果](gallery.html)。完整原始数值及每张脸的源/目标 EAR、MAR、眉部几何、旋转矩阵见 [reviewed-results.json](artifacts/run-context/reviewed-results.json)，机器可读汇总见 [metrics.csv](artifacts/run-context/metrics.csv)。CSV 的 utility 列为自动测量阶段 indeterminate，独立复核以 reviewed-results.json 为准。

## 指标定义与适用边界

- Identity：OpenCV SFace 2021dec + YuNet 五点对齐，cosine ∈ [-1,1]，负数正常。每个有效样本原图与自身的控制相似度约 1。没有建立身份图库，也没有多识别器攻击测试。
- Eye：每侧两组眼睑距离的均值除以眼角距离（EAR）；报告原始值、输出值和绝对差。Mouth：内唇 13–14 距离除以 78–308 嘴宽（MAR）。眉部：眉眼距离除以眼宽。它们是连续几何量，不等于闭眼/清醒/吞咽真值。
- Pose：六个关键点 + 通用 3D 模板，SQPNP 后 LM 优化，假设焦距=图宽；比较旋转矩阵相对旋转角。正深度及重投影残差用于基本质量门禁，但模板/焦距未校准，侧脸和低清图的误差仍可很大。裁剪和对齐会影响估计。
- 本轮使用旧版 FaceMesh，与官方模型条件一致；没有输出 blendshape、expression embedding、FACS AU 或 gaze 数值。眉部/眼嘴几何只是 expression/state 的部分代理。
- 独立 utility 接口及 downstream consistency 接口见 interfaces.py。独立视觉复核实现见 utility_review.py。没有调用远端 AI；没有真实下游护理模型，因此 downstream 一律 not_implemented。

## 失败案例与独立 utility 证据

1. **闭嘴变露齿笑**：s01 与 s12。s01 的 ΔMAR 仅约 0.0225，却出现可见牙齿和笑容，说明单一 landmark 阈值会漏掉重要语义变化。
2. **不可见状态被编造**：s06 原始眼嘴受到眼镜、伞柄、胡须遮挡，输出生成可见眼睛和张嘴。不能把生成内容当作原始状态补全。
3. **侧脸/眼镜与姿态失真**：s03 眼嘴几何变化大，近似姿态差约 18.75°；s04 无法提取输入关键点；s08 男性侧脸失败。
4. **低清/小脸**：s07 关键点失败；s10 部分脸框过小；s02 主体生成脸严重失真，输出状态验证失败。
5. **身份不可测**：s10/f2 不能得到有效单脸身份分数。不是隐私成功。
6. **低照度/旋转/遮挡漏检**：s13/s14/s15 无候选。没有回退为原图输出。
7. **姿态估计自身不稳定**：第一轮紧裁剪出现约 171.68° 相对角异常；对照轮改为 SQPNP+LM 及重投影检查。它说明 validator 也需要独立标注校准，不能用原始估计直接判断模型。

## 是否进入 Phase 1

**NO-GO：暂不进入完整 FastAPI/OCR/service 开发。** 已证明本地推理可行，但尚未证明护理 utility；而且已有明确反例。

下一步应继续 Phase 0：先验证官方 RetinaFace/align=True 与当前 crop 的受控差异；建立获授权的闭眼、进食遮挡、卧床/侧卧、低照度真实数据；邀请护理专家做眼嘴/表情/姿态及“不可判断”标注；用独立状态模型与本地 downstream AI 做配对一致性评估；再比较其他可替换 anonymizer。模型/数据许可与身份阈值校准仍须解决。

7 项针对零脸策略、未知评估不通过、视觉失真拒绝、结果完整性与 identity self-control 的回归检查通过，见 artifacts/tests.log。它们仅验证 spike 的证据语义，不证明模型达到业务标准。

## 研究来源

- [GANonymization 官方仓库](https://github.com/hcmlab/GANonymization)
- [GANonymization 论文](https://d-nb.info/1358397392/34)
- [SFace 模型与许可](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)
- [YuNet 模型与许可](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- [CelebA 条款](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
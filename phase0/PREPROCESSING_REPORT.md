# Phase 0：官方预处理对照实验

**结论：本地研究推理可行；护理 utility 尚未通过，维持 NO-GO，不进入 Phase 1。**

本报告接续 [初始报告](REPORT.md)。2026-09-29 整理已完成的 `preprocessing-v1` 实验，没有重新训练模型，也没有搭建 FastAPI、OCR 或服务架构。官方 RetinaFace 裁剪/对齐对照已从待办变为完成；真实护理场景验收仍未完成。

## 数据与控制变量

16 张输入全部执行：12 张公开许可照片 + 4 张人工压力变换，来源见 [署名与许可](ATTRIBUTION.md)。包含相关人物、同源变换，不能按 16 个独立临床样本统计。没有真实卧床/侧卧照片，闭眼、真实低照度、进食遮挡也缺少可靠真值。

三个实验组使用相同输入文件、同一官方 generator 权重、同一 SFace 和 FaceMesh validator、seed=42、CPU 4 线程。YuNet 组的 13 张候选与上一轮对应文件 SHA-256 全部相同，基线未漂移。

| 组 | 检测/预处理 | 检测框 | 生成候选 | identity 可测 | eye/mouth/pose 全部可测 | 两类均可测 |
|---|---|---:|---:|---:|---:|---:|
| YuNet context | 阈值 0.6，1/0.5 尺度，每侧 20% margin，32px 门限 | 20 | 13 | 12 | 12 | 11 |
| RetinaFace unaligned | 阈值 0.9，默认 upscaling，官方 FaceCrop，align=False | 15 | 8 | 8 | 8 | 8 |
| RetinaFace aligned | 同一批检测框，官方 FaceCrop，align=True | 15 | 7 | 7 | 7 | 7 |

检测框数不等于真实人数，生成数不等于匿名化通过数，全部候选禁止放行。YuNet 与 RetinaFace 同时改变了检测器、裁剪边界和尺寸规则，不能把差异只归因于对齐；只有后两组在相同检测结果上比较对齐开关。跨检测器以原图 bbox 的互为最佳 IoU 配对，含糊配对保留未匹配，不能直接对应 f 编号。

调用未修改的上游 FaceCrop 实现；为了两组共享同一次检测结果，仅临时向其检测调用注入缓存的真实检测记录。未对齐裁剪已与原图 bbox 切片逐像素核对。本轮是官方组件的最小推理适配，仍不声称完整官方 CLI 环境原样复现。

## 实际环境与 checkpoint

- 硬件/系统：Apple M1 Pro，macOS 15.3.2 arm64；均使用 CPU。
- 生成环境：Python 3.8.20、torch 2.2.2、torchvision 0.17.2、MediaPipe 0.10.9、NumPy 1.24.4、OpenCV contrib 4.8.1.78。完整锁定见 [生成环境依赖](requirements.lock.txt)。MediaPipe 在 macOS 仍需要本地图形上下文。
- 独立预处理环境：Python 3.8.20、tensorflow-macos 2.13.0、Keras 2.13.1、retina-face 0.0.13、NumPy 1.24.3、OpenCV 4.8.1.78。见 [预处理依赖](requirements-retina.lock.txt)。这是研究脚本之间通过本地文件交换结果，不是已实现的服务 inference worker。
- GANonymization 源代码 commit：`7dbe45e3b172670ff59be982c662749165ab5ced`。
- GAN checkpoint：[官方论文版下载](https://mediastore.rz.uni-augsburg.de/get/NsLjQYey65/)，687127579 bytes，epoch=24，global_step=283800。
- GAN checkpoint SHA-256：`eba49bd525033b55022a91d6b4398ab088b51339fd6895db7afdd648d776eec9`。
- RetinaFace 权重：[retinaface.h5](https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5)，118667368 bytes。
- RetinaFace SHA-256：`ecb2393a89da3dd3d6796ad86660e298f62a0c8ae7578d92eb6af14e0bb93adf`。
- SFace SHA-256：`0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79`。

这些哈希用于复核本地文件，不是发行方签名或商用许可证明。模型与训练数据许可尚未解决。

## 逐图结果与指标

- [16 张输入的三组图片对照（本地）](gallery-preprocessing.html)：逐脸 source/candidate；source 是对应组预处理后的裁剪。
- [逐图/逐脸 CSV](evidence/preprocessing-v1/metrics.csv)：包含未检出、未生成、不可测记录；空值不能当作 0 或通过。
- [完整数值证据 JSON](evidence/preprocessing-v1/measurements.json)：保留输入/候选哈希、原始及生成 EAR/MAR/眉部比值、旋转矩阵、空间配对、失败原因、环境与代码哈希。原始实验文件的 SHA-256 也已记录。

identity 是 SFace cosine，未校准业务阈值。eye 是 MediaPipe 33/263 两侧 EAR 绝对差，mouth 是内唇开口/嘴宽 MAR 绝对差；这些几何量不等于闭眼、吞咽或疼痛真值。Pose 是通用六点模板估算的相对旋转角，仅比较**经过该组裁剪/对齐之后**的输入与生成脸；未评估整图原始头姿及逆变换回贴。

表情只有眉部/眼嘴几何代理，未实现独立表情模型、blendshapes 或临床表情评分。不能把“所有数值可测”视为 utility preservation。没有预设通过阈值，identity/state pass 均为 null。

## 失败分类与观察

| 类别 | 本轮证据 | 处理 |
|---|---|---|
| 已知有人脸却零检测 | 三组均漏检 s13/s14/s15；RetinaFace 另未检出 s01/s07 | fail closed，不返回原图作为成功结果 |
| 输入关键点失败 | YuNet 5 张检测脸，RetinaFace 未对齐 7 张、对齐 8 张 | 不生成候选 |
| 小脸不足 | YuNet 2 个框不足 32px | 拒绝；该门限未额外加入官方 RetinaFace 组 |
| 验证不可测 | YuNet 13 个生成候选中 2 个不能同时完成两类测量 | 保留不可判定，禁止放行 |
| 状态/可见性变化 | s10 RetinaFace f1 输入向下看的眼部状态与生成前视睁眼外观不一致 | 表明小几何误差仍可能遗漏语义变化 |
| 侧脸/遮挡失真 | s02 未对齐前景侧脸候选失真；对齐后该脸关键点失败 | 不能以较低 identity 判定成功 |
| 低照度下生成细节 | s16 候选出现更清楚明亮的面部细节 | 不能当作恢复的真实护理信息 |

上述视觉描述是研究观察，不是护理专家或独立状态模型验收；没有给新图自动套用上一轮主观标签。s12 在两种 RetinaFace 裁剪下粗看保留了闭嘴，比原 YuNet 候选露齿笑更好，说明预处理值得研究，但个别改善不足以支持整体通过。

生成肤色、嘴唇颜色、皮肤纹理、皱纹等均不能默认代表真实护理信息。眼镜、遮挡物消失可能改变可判断性，同样需要验证。

## 独立 utility 与下一阶段门槛

新增 hash-bound 标注接口，将独立观察绑定到确切 source/candidate 文件，拒绝复用错误候选的标注。眼/嘴/表情/头姿均记录状态与可见性；原图不可观察而生成图给出明确状态会记录证据缺失。空标注、未知状态及缺少 downstream 评估均不得产生通过结论。见 [标注说明](UTILITY_EVALUATION.md)。

当前只有接口和空白待评审模板；没有伪造专家标签，没有执行真实 downstream AI。SFace 单识别器、未经校准的身份阈值也不足以证明隐私安全。

**继续 Phase 0 定向研究，不开始 Phase 1。** 下一轮应取得获授权的真实护理样本与独立状态/可见性标注，校准身份与 utility 阈值，接入本地下游护理模型做配对一致性评估；如 baseline 仍不能保留状态，再经研究比较替换 anonymizer。公开样例冒烟测试完成不等于老人护理场景验收通过。

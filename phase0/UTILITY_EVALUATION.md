# 独立 utility 标注协议（Phase 0）

`annotation_utility.py` 提供独立于 FaceMesh 数值的标注校验接口；不自动生成观察结论，不替代护理专家或 downstream AI。

1. 运行 `export_preprocessing.py` 生成本地 `artifacts/preprocessing-v1/review-template.json`，每个候选绑定 source/candidate SHA-256。
2. 由评审者分别观察 source 和 candidate，填写眼状态、嘴状态、表情、粗头姿以及每项可见性；无法判断必须记 uncertain。尽可能打乱顺序并隐藏自动指标，避免数值影响判断。source 是预处理裁剪，整图姿态另需评估。
3. 填写 reviewer 与 review_type（research_observer 或 care_expert），保留备注。研究观察者不得自称专家。不要改写或补全原图不可见的状态。
4. 调用 `evaluate_annotations(review, source_sha256, candidate_sha256)`。图像哈希必须与当前文件一致。已知状态改变或缺少原始证据却生成确定状态会失败；缺失、不可判定或仅粗标注一致均不能通过临床验收。
5. 将完成的评审存入新的本地文件，不覆盖空白模板或原始实验。导出脚本会重新写入空白模板，已完成评审不要使用其文件名。

当前四类标注是粗粒度接口：尚不覆盖分眼精细评分、gaze、疼痛强度或吞咽；后续应由业务专家确定量表、双人复核和分歧处理。新候选不得仅按 sample id 自动继承旧结论。

DownstreamConsistencyEvaluator 保留在 interfaces.py，尚未实现。真实下游评估应固定模型版本及配置，对同一输入/输出比较护理风险判断、置信度及不可判定状态，并记录遮挡、场景、头姿和图像来源。阈值与可接受误差必须经过独立业务校准，不能以本次公开样例拟合后直接作为验收标准。

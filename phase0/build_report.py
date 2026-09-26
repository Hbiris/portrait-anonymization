"""Build an offline, self-contained gallery and evidence-backed Phase 0 report."""
import base64
import collections
import html
import io
import json
import pathlib
from PIL import Image

ROOT=pathlib.Path(__file__).resolve().parent
RUN=ROOT/'artifacts/run-context'

def thumb(path, size=(460,380)):
    with Image.open(path) as original:
        pic=original.convert('RGB');pic.thumbnail(size);buffer=io.BytesIO();pic.save(buffer,format='JPEG',quality=88)
    return 'data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode()

def number(value):return 'N/A' if value is None else '%.4f'%value

def main():
    rows=json.loads((RUN/'reviewed-results.json').read_text());env=json.loads((RUN/'environment.json').read_text());ckpt=json.loads((RUN/'checkpoint.json').read_text())
    cards=[];table=[]
    for row in rows:
        parts=['<section><h2>'+row['id']+' · '+html.escape(row['status'])+'</h2>',
               '<p>'+html.escape(row['title'])+'</p><p>公开照片 / '+('人工压力测试：'+row.get('transformation','') if row['transformed'] else '自然拍摄')+'</p>',
               '<details><summary>完整原始输入（公开样例，仅本地）</summary><img src="'+thumb(ROOT/row['path'])+'"></details>',
               '<p>检测结果：'+html.escape(str(row['detectors']))+'；utility：'+row['utility_status']+'；禁止对外放行</p>']
        for face in row['faces'] or [{}]:
            state=face.get('state',{});delta=state.get('absolute_delta',{});identity=face.get('identity',{});u=face.get('utility',{})
            id=row['id']+('/f'+str(face['index']) if 'index' in face else '')
            vals=[id,number(identity.get('identity_similarity')),number(delta.get('ear_33_side')),number(delta.get('ear_263_side')),number(delta.get('mouth_aperture_ratio')),number(state.get('head_pose_difference_degrees')),face.get('failure',face.get('status',row['status'])),u.get('status','indeterminate')]
            table.append('| '+' | '.join(vals)+' |')
            parts.append('<h3>'+id+'</h3>')
            if face.get('artifacts'):
                parts.append('<div class="pair">')
                for key,label in [('source','原始人脸裁剪'),('condition','官方 478 点条件图'),('candidate','未通过验收的生成候选')]:
                    parts.append('<figure><img src="'+thumb(ROOT/face['artifacts'][key],(300,300))+'"><figcaption>'+label+'</figcaption></figure>')
                parts.append('</div><p>identity cosine='+vals[1]+'；EAR 两侧差='+vals[2]+' / '+vals[3]+'；MAR 差='+vals[4]+'；近似姿态差='+vals[5]+'°</p>')
                parts.append('<p class="warning">独立视觉复核：'+html.escape(u.get('evidence','未复核'))+'</p>')
            else:parts.append('<p class="warning">无生成结果：'+html.escape(vals[6])+'。未用原图替代失败结果。</p>')
        parts.append('<p><a href="'+html.escape(row['source_page'],quote=True)+'">来源及许可</a> · '+html.escape(row['license'])+'</p></section>');cards.append(''.join(parts))
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Phase 0 · GANonymization 本地评估</title>
<style>body{font:16px/1.7 system-ui,sans-serif;max-width:1120px;margin:32px auto;padding:0 24px;background:#f4f5f7;color:#17222d}h1,h2{line-height:1.3}section{background:white;border:1px solid #d7dde3;border-radius:12px;padding:24px;margin:24px 0}.pair{display:flex;flex-wrap:wrap;gap:16px}figure{margin:0}figure img{width:280px;max-width:100%}figcaption{font-size:14px}img{max-width:100%}.warning{color:#983b22}a{color:#125c9b}header{background:#172b3a;color:white;padding:28px;border-radius:12px}</style>
<header><h1>Phase 0：技术可运行，护理 utility 未通过</h1><p>16 张输入 · 12 张公开照片 + 4 张人工压力测试 · 13 张生成脸候选 · 0 张获准放行</p><p>这些是研究中间结果，不是可上传的匿名护理图片。肤色、纹理、皱纹、唇色均为合成内容。</p></header>
<p>本页图片已内嵌，无外部模型或图片请求。视觉复核由 Codex 完成，独立于关键点计算，但不是护理专家标注，也不是经过验证的下游评估。</p>'''+''.join(cards)+'</html>'
    (ROOT/'gallery.html').write_text(document)
    lines=['# Phase 0 实测报告',
    '\n结论：**官方 checkpoint 已在本机运行；当前 baseline 不值得直接进入完整 Phase 1 服务开发。** 先继续定向模型/预处理评估和真实业务数据验收。没有创建 FastAPI、OCR 或服务架构。',
    '\n## 实际环境与复现范围',
    '\n- macOS 15.3.2，arm64；MediaPipe 日志识别 Apple M1 Pro。Python 3.8.20（项目隔离环境）。',
    '- torch 2.2.2、torchvision 0.17.2、MediaPipe 0.10.9、OpenCV contrib 4.8.1.78、NumPy 1.24.4、Pillow 10.4.0、Lightning 2.0.9、torchmetrics 1.2.1、SciPy 1.10.1。完整依赖见 [requirements.lock.txt](requirements.lock.txt)。',
    '- 生成器与 OpenCV 检测/识别使用 CPU，torch 4 线程；MediaPipe CPU graph 仍需 macOS GL 上下文，沙箱内创建失败，放到本机隔离环境后成功。没有 CUDA，也未把 MPS 用于本轮生成推理。',
    '- 官方源代码 commit：`'+env['upstream_commit']+'`。使用未修改的 GeneratorUNet、ZeroPaddingResize、FacialLandmarks478，严格加载 generator state_dict。',
    '- **不是官方完整 CLI 的原样复现**：为避免研究训练/GUI/TensorFlow 全依赖，使用最小推理适配器；YuNet 替代 RetinaFace 裁剪。对照轮使用 1.0/0.5 尺度检测、NMS、bbox 每侧 20% 上下文、align=False。没有改模型权重或训练。该差异会影响质量，因此不能把所有失败归因于生成器。',
    '- 不做整图回贴：官方生成器输出 512×512 人脸裁剪候选，背景是生成模型的黑底；完整身体/场景保持、整图合成与全图覆盖验证尚未实现。Phase 0 的图片交付是逐脸候选及逐输入失败记录，不是服务可用整图。',
    '\n## Checkpoint',
    '\n- 官方 README 的论文版 / 25 epochs 链接：[University of Augsburg](https://mediastore.rz.uni-augsburg.de/get/NsLjQYey65/)。',
    '- 本地：`models/publication-download`；'+str(ckpt['bytes'])+' bytes（约 655.30 MiB）。',
    '- SHA-256：`'+ckpt['sha256']+'`。这是本地完整下载后的哈希，不是发行方数字签名。',
    '- checkpoint epoch=24（零基，即第 25 个 epoch），global_step=283800；训练 Lightning=2.0.0。超参数 n_epochs=50 是训练计划值，不代表这个权重已训练 50 epochs。',
    '- 生成器 54,404,099 个参数；用 weights_only=True 读取 checkpoint，不加载优化器用于推理。',
    '- 主仓库 MIT 不等于预训练模型商用授权已解决；其 CelebA 训练来源的限制仍待解决。当前仅研究 baseline。',
    '\n## 数据与评估范围',
    '\n16 张输入：12 张公开许可照片、4 张派生压力测试。12 张不等于 12 个独立人物：s03/s04 是同一个人的不同照片，s13–s16 与父样本相关。来源、作者、许可和哈希见 [ATTRIBUTION.md](ATTRIBUTION.md) 与 data/sources.json。',
    '\n| 场景 | 覆盖与缺口 |\n|---|---|\n| 老年外观人像 | 多张公开照片；没有年龄/护理状态真值，不能凭标题确认年龄 |\n| 眼镜、侧脸 | s03/s04、s08、s11 等 |\n| 张嘴、进食环境 | s08/s09/s10；不等于已验证吞咽或真实食物遮嘴 |\n| 闭眼 | s06 是睡眠候选，眼镜/胡须/伞遮挡；s09 向下看，不能作为经标注的闭眼真值 |\n| 低照度 | s13/s16 是亮度乘 0.18 的人工变换，不能替代真实低照度噪声 |\n| 遮挡 | 自然手部、眼镜等遮挡；s15 是人工矩形遮挡，不能替代真实食物/器械遮挡 |\n| 卧床/侧卧 | **无真实样本**；s14 仅图像旋转 90°，不是侧卧身体/面部视角变化 |',
    '\n## 实测汇总',
    '\n- 16 张全部执行并有结果记录；11 张输入产生共 13 张生成脸候选。',
    '- YuNet 共提出 20 个人脸候选（包含小脸、背景图像和潜在误检，不能当成人工确认人数）。',
    '- 12 个生成脸有 SFace identity cosine；12 个有眼嘴/近似姿态指标；11 个同时有两类指标。',
    '- 5 个人脸因输入关键点提取失败未生成；2 个因脸框小于 32 像素拒绝。',
    '- s13/s14/s15 两检测器均返回零脸，但根据已知原图及变换可知仍有人脸，属于检测漏检案例。**多个 detector 均未检出不等于确认没有人脸。** 当前 zero_face_policy=reject；接口支持 continue_if_all_clear，但任何怀疑/错误必须拒绝，该策略也不能绕过已知图像证据。',
    '- 独立于关键点计算的 Codex 视觉复核：13 张候选中 9 张发现明确状态/可见性/表情失真，4 张无法确认。它是可用性反例证据，不是护理专家验收或独立模型验证。',
    '- 当前没有经业务校准的阈值，所有 identity/state pass 均为 null。低相似度不等于匿名化通过，特别是生成脸失真时。0 张获准放行。',
    '- 对照轮样本处理累计约 15.41 秒（不含模型加载、下载、报告生成），含失败快速返回；不是吞吐或 SLA 基准。',
    '\n## 每张输入/每张脸的指标',
    '\nN/A 表示未得到可靠结果，绝不按 0 或通过处理。face 编号是检测排序，不是身份。表中 eye A/B 指 MediaPipe 33/263 两侧，不命名解剖学左右眼以避免镜像混淆。',
    '\n| 输入/脸 | identity cosine | ΔEAR A | ΔEAR B | ΔMAR | 近似姿态差 ° | 执行状态 | 独立 utility 复核 |',
    '|---|---:|---:|---:|---:|---:|---|---|']+table+[
    '\n[浏览全部 16 张输入与候选/失败结果](gallery.html)。完整原始数值及每张脸的源/目标 EAR、MAR、眉部几何、旋转矩阵见 [reviewed-results.json](artifacts/run-context/reviewed-results.json)，机器可读汇总见 [metrics.csv](artifacts/run-context/metrics.csv)。CSV 的 utility 列为自动测量阶段 indeterminate，独立复核以 reviewed-results.json 为准。',
    '\n## 指标定义与适用边界',
    '\n- Identity：OpenCV SFace 2021dec + YuNet 五点对齐，cosine ∈ [-1,1]，负数正常。每个有效样本原图与自身的控制相似度约 1。没有建立身份图库，也没有多识别器攻击测试。',
    '- Eye：每侧两组眼睑距离的均值除以眼角距离（EAR）；报告原始值、输出值和绝对差。Mouth：内唇 13–14 距离除以 78–308 嘴宽（MAR）。眉部：眉眼距离除以眼宽。它们是连续几何量，不等于闭眼/清醒/吞咽真值。',
    '- Pose：六个关键点 + 通用 3D 模板，SQPNP 后 LM 优化，假设焦距=图宽；比较旋转矩阵相对旋转角。正深度及重投影残差用于基本质量门禁，但模板/焦距未校准，侧脸和低清图的误差仍可很大。裁剪和对齐会影响估计。',
    '- 本轮使用旧版 FaceMesh，与官方模型条件一致；没有输出 blendshape、expression embedding、FACS AU 或 gaze 数值。眉部/眼嘴几何只是 expression/state 的部分代理。',
    '- 独立 utility 接口及 downstream consistency 接口见 interfaces.py。独立视觉复核实现见 utility_review.py。没有调用远端 AI；没有真实下游护理模型，因此 downstream 一律 not_implemented。',
    '\n## 失败案例与独立 utility 证据',
    '\n1. **闭嘴变露齿笑**：s01 与 s12。s01 的 ΔMAR 仅约 0.0225，却出现可见牙齿和笑容，说明单一 landmark 阈值会漏掉重要语义变化。',
    '2. **不可见状态被编造**：s06 原始眼嘴受到眼镜、伞柄、胡须遮挡，输出生成可见眼睛和张嘴。不能把生成内容当作原始状态补全。',
    '3. **侧脸/眼镜与姿态失真**：s03 眼嘴几何变化大，近似姿态差约 18.75°；s04 无法提取输入关键点；s08 男性侧脸失败。',
    '4. **低清/小脸**：s07 关键点失败；s10 部分脸框过小；s02 主体生成脸严重失真，输出状态验证失败。',
    '5. **身份不可测**：s10/f2 不能得到有效单脸身份分数。不是隐私成功。',
    '6. **低照度/旋转/遮挡漏检**：s13/s14/s15 无候选。没有回退为原图输出。',
    '7. **姿态估计自身不稳定**：第一轮紧裁剪出现约 171.68° 相对角异常；对照轮改为 SQPNP+LM 及重投影检查。它说明 validator 也需要独立标注校准，不能用原始估计直接判断模型。',
    '\n## 是否进入 Phase 1',
    '\n**NO-GO：暂不进入完整 FastAPI/OCR/service 开发。** 已证明本地推理可行，但尚未证明护理 utility；而且已有明确反例。',
    '\n下一步应继续 Phase 0：先验证官方 RetinaFace/align=True 与当前 crop 的受控差异；建立获授权的闭眼、进食遮挡、卧床/侧卧、低照度真实数据；邀请护理专家做眼嘴/表情/姿态及“不可判断”标注；用独立状态模型与本地 downstream AI 做配对一致性评估；再比较其他可替换 anonymizer。模型/数据许可与身份阈值校准仍须解决。',
    '\n7 项针对零脸策略、未知评估不通过、视觉失真拒绝、结果完整性与 identity self-control 的回归检查通过，见 artifacts/tests.log。它们仅验证 spike 的证据语义，不证明模型达到业务标准。',
    '\n## 研究来源',
    '\n- [GANonymization 官方仓库](https://github.com/hcmlab/GANonymization)\n- [GANonymization 论文](https://d-nb.info/1358397392/34)\n- [SFace 模型与许可](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface)\n- [YuNet 模型与许可](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)\n- [CelebA 条款](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)']
    (ROOT/'REPORT.md').write_text('\n'.join(lines))

if __name__=='__main__':main()

"""Export existing measurements; never rerun inference or infer review labels."""
import csv
import html
import json
from collections import Counter
from pathlib import Path
from annotation_utility import review_template
from experiment_utils import sha256

ROOT = Path(__file__).resolve().parent


def export(name='preprocessing-v1'):
    raw = ROOT/'artifacts'/name/'experiment.json'
    experiment = json.loads(raw.read_text())
    target = ROOT/'evidence'/name
    target.mkdir(parents=True, exist_ok=True)
    evidence = dict(experiment, raw_experiment_sha256=sha256(raw))
    evidence['summary'] = {}
    rows, reviews, sections = [], [], []
    for arm in experiment['arms']:
        faces = [f for s in experiment['samples'] for f in s['arms'][arm]]
        evidence['summary'][arm] = {
            'inputs': len(experiment['samples']), 'detections':len(faces),
            'generated':sum('candidate_sha256' in f for f in faces),
            'identity_measured':sum(f.get('identity',{}).get('status')=='measured_uncalibrated' for f in faces),
            'state_measured':sum(f.get('state',{}).get('status')=='measured_uncalibrated' for f in faces),
            'both_measured':sum(f['status']=='candidate_measured' for f in faces),
            'failures':dict(Counter(f['failure'] for f in faces if 'failure' in f)),
            'zero_detection_inputs':[s['id'] for s in experiment['samples'] if not s['arms'][arm]],
            'release_allowed':False}
    for sample in experiment['samples']:
        sections.append('<section><h2>'+sample['id']+'</h2><p>各组 f 编号仅为组内检测顺序，不能跨组按编号认定同一张脸。坐标配对见 JSON。</p><div class="arms">')
        for arm in experiment['arms']:
            sections.append('<article><h3>'+arm+'</h3>')
            faces = sample['arms'][arm]
            for face in faces or [{}]:
                state = face.get('state',{})
                delta = state.get('absolute_delta',{})
                row = {'sample':sample['id'],'arm':arm,'face':face.get('index'),
                       'status':face.get('status','zero_detections_known_face_no_release'),
                       'failure':face.get('failure',''),
                       'identity_cosine':face.get('identity',{}).get('identity_similarity'),
                       'delta_ear_33':delta.get('ear_33_side'),
                       'delta_ear_263':delta.get('ear_263_side'),
                       'delta_mar':delta.get('mouth_aperture_ratio'),
                       'pose_difference_degrees':state.get('head_pose_difference_degrees'),
                       'source_sha256':face.get('source_sha256'),
                       'candidate_sha256':face.get('candidate_sha256'),
                       'release_allowed':False}
                rows.append(row)
                sections.append('<p>'+html.escape('f%s: %s %s' % (row['face'],row['status'],row['failure']))+'</p>')
                if face.get('candidate_sha256'):
                    for kind in ('source','candidate'):
                        path = ROOT/face['artifacts'][kind]
                        if sha256(path)!=face[kind+'_sha256']:
                            raise ValueError('Artifact mismatch: '+str(path))
                    reviews.append(review_template(sample['id'],arm,face['index'],face['source_sha256'],face['candidate_sha256']))
                    sections.append('<div class="pair">'+''.join('<figure><img loading="lazy" src="'+html.escape(face['artifacts'][k],quote=True)+'"><figcaption>'+k+'</figcaption></figure>' for k in ('source','candidate'))+'</div>')
                    sections.append('<p>'+html.escape('identity={}; ΔEAR={}/{}; ΔMAR={}; pose°={}'.format(row['identity_cosine'],row['delta_ear_33'],row['delta_ear_263'],row['delta_mar'],row['pose_difference_degrees']))+'</p>')
            sections.append('</article>')
        sections.append('</div></section>')
    # Retain all measured source/output values, hashes, spatial matches and failures.
    (target/'measurements.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    with (target/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (ROOT/'artifacts'/name/'review-template.json').write_text(json.dumps(reviews,indent=2)+'\n')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Phase 0 预处理对照</title><style>body{font-family:system-ui;margin:24px;background:#f5f7fa;color:#182131}h1{font-size:26px}.arms{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}article{padding:12px;background:white;border:1px solid #ddd;overflow-wrap:anywhere}.pair{display:flex}figure{margin:3px;width:50%}img{width:100%}p{font-size:13px}section{margin-top:36px}@media(max-width:900px){.arms{grid-template-columns:1fr}}</style><h1>Phase 0 · 16 张公开样例，三组预处理对照</h1><p>研究候选全部禁止放行。source 是各组预处理后的裁剪，不是整图。数值未经业务阈值校准；null / None 为不可测。生成肤色、纹理和皱纹不能用作真实护理信息。</p><p><a href="PREPROCESSING_REPORT.md">报告</a> · <a href="ATTRIBUTION.md">来源与许可</a> · <a href="evidence/preprocessing-v1/metrics.csv">逐脸指标</a></p>'''+''.join(sections)+'</html>'
    page = page.replace('evidence/preprocessing-v1/', 'evidence/'+html.escape(name, quote=True)+'/')
    (ROOT/'gallery-preprocessing.html').write_text(page)
    print(json.dumps(evidence['summary'],indent=2))


if __name__=='__main__':
    export()

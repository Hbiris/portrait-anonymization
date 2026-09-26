"""Add explicitly synthetic stress cases; do not call rotation real bed rest."""
import hashlib
import json
import pathlib
import cv2
import numpy as np

ROOT=pathlib.Path(__file__).resolve().parent
rows=json.loads((ROOT/'data/sources.json').read_text())
for r in rows:
    # Two photographs of the same man; not independent identities.
    if r['id']=='s04':r['source_group']='s03'
    r['scene_coverage']='public_photo_not_clinical_ground_truth'
stress=[('s13','s01','low_light'),('s14','s12','roll_90'),('s15','s01','lower_face_occlusion'),('s16','s08','low_light')]
(ROOT/'data/stress').mkdir(exist_ok=True)
for id,parent,operation in stress:
    base=next(r for r in rows if r['id']==parent)
    image=cv2.imread(str(ROOT/base['path']))
    if operation=='low_light':image=np.clip(image.astype(float)*.18,0,255).astype(np.uint8)
    elif operation=='roll_90':image=cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
    elif operation=='lower_face_occlusion':
        h,w=image.shape[:2];cv2.rectangle(image,(int(.2*w),int(.63*h)),(int(.85*w),int(.87*h)),(45,45,45),-1)
    path=ROOT/('data/stress/'+id+'.png');cv2.imwrite(str(path),image)
    row=dict(base);row.update(id=id,path=str(path.relative_to(ROOT)),parent=parent,source_group=base['source_group'],
                            transformed=True,transformation=operation,tags=['synthetic_stress',operation],
                            title=base['title']+' / synthetic '+operation,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    rows.append(row)
(ROOT/'data/samples.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))

"""Independent visual-review evidence ingestion, separate from MediaPipe metrics.

Reviews here are Codex visual observations on local contact sheets, not clinician
annotations or an independent validated medical model. They may falsify a utility
claim, but cannot establish clinical utility. Downstream evaluation is still TODO.
"""
import json
import pathlib

ROOT=pathlib.Path(__file__).resolve().parent
OBSERVATIONS={
    's01:0': ('fail', '原图嘴部闭合，输出出现露齿笑；眼睛朝向也有可见变化。即使 EAR/MAR 差较小，也不能认为面部状态保留。', ['mouth_state','facial_expression']),
    's02:0': ('indeterminate', '背景人脸分辨率低；生成结果明显改变外观，无法可靠复核眼嘴细节。', []),
    's02:2': ('fail', '输出严重模糊、五官变形；嘴旁原有手部遮挡未被保留，输出关键点验证也失败。', ['mouth_state','facial_expression']),
    's03:0': ('fail', '眼镜消失；侧脸轮廓和眼嘴明显变形，姿态估计差约 18.75 度。', ['eye_state','mouth_state','head_pose']),
    's05:0': ('indeterminate', '大致头部朝向接近，但嘴部外观和整体表情有变化，不能确认护理语义等价。', []),
    's06:0': ('fail', '源图眼镜、伞柄和胡须遮挡眼嘴；输出生成清晰眼睛和张开的嘴，不能将生成状态视作原始证据。', ['eye_state','mouth_state']),
    's08:0': ('fail', '原图眼睛可见且嘴部张开，输出眼睑和嘴形改变；眼镜消失。', ['eye_state','mouth_state','facial_expression']),
    's09:0': ('indeterminate', '低头和微笑大致可见；输出一侧眼睑状态不清晰，单张图不能区分闭眼与向下看。', []),
    's10:2': ('fail', '原图低头，输出变成明显侧脸且变形；身份验证无法得到有效单脸。', ['head_pose','facial_expression']),
    's10:3': ('fail', '原图低头且眼睑下垂，输出眼部更清晰朝前、嘴角上扬；不能据此推断清醒或情绪状态。', ['eye_state','facial_expression']),
    's11:0': ('indeterminate', '原图眼镜造成眼部难辨，输出去除了眼镜并合成眼睛，无法确认眼睛真实状态被保留。', []),
    's12:0': ('fail', '原图嘴部闭合、表情严肃，输出出现露齿笑。', ['mouth_state','facial_expression']),
    's16:0': ('fail', '原图为人工变暗的进食场景；输出变亮并改变嘴形，生成细节不能补充真实护理证据。', ['mouth_state','facial_expression']),
}

class IndependentVisualReviewValidator:
    def validate(self, original, candidate, context):
        key=context['sample_id']+':'+str(context['face_index'])
        status,note,dimensions=OBSERVATIONS.get(key,('indeterminate','无独立视觉复核记录。',[]))
        return {'status':status,'pass':False if status=='fail' else None,
                'method':'Codex visual inspection of local source/candidate pairs, independent of landmark computation',
                'reviewer':'Codex; not a clinician; no clinical ground truth',
                'evidence':note,'observed_changed_dimensions':dimensions,
                'downstream_consistency':{'status':'not_implemented','pass':None},
                'clinical_utility_status':'not_established','synthetic_appearance_is_care_evidence':False}

def main():
    path=ROOT/'artifacts/run-context/results.json';rows=json.loads(path.read_text());validator=IndependentVisualReviewValidator()
    for row in rows:
        for face in row['faces']:
            if face.get('artifacts'):
                face['utility']=validator.validate(None,None,{'sample_id':row['id'],'face_index':face['index']})
        row['release_allowed']=False
        row['utility_status']='fail' if any(f.get('utility',{}).get('status')=='fail' for f in row['faces']) else 'indeterminate'
    (path.parent/'reviewed-results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=='__main__':main()

"""Evaluate three preprocessing arms with one fixed generator and validators.

Run only after retina_preprocess.py. No past subjective review is copied to new
candidates: a review must identify the exact candidate hash. All release flags
remain false. Pose metrics are within each preprocessed crop, not the original
full-image coordinate frame; this experiment does not test inverse composition.
"""
import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import cv2
import torch

from experiment_utils import (match_boxes, new_run_directory, sha256,
                              state_measurement_status, write_json)
from spike import GANonymization, Detector, SFaceValidator, StateValidator, contact_sheet

ROOT = Path(__file__).resolve().parent
ARMS = ('yunet_context', 'retina_unaligned', 'retina_aligned')


def evaluate_face(generator, identity, state, crop, directory, stem, record):
    started = time.time()
    record.update(release_allowed=False, utility={'status':'indeterminate',
                  'independent_review':'pending_for_this_exact_candidate',
                  'downstream_consistency':'not_implemented', 'pass':None})
    try:
        if crop is None or crop.size == 0:
            raise ValueError('empty_preprocessing_crop')
        source, candidate, condition = generator.anonymize(crop)
        record['artifacts'] = {}
        for key, image in [('source',source),('condition',condition),('candidate',candidate)]:
            path = directory/(stem+'-'+key+'.png')
            if not cv2.imwrite(str(path),image):
                raise IOError('artifact_write_failed')
            record['artifacts'][key] = str(path.relative_to(ROOT))
        record['source_sha256'] = sha256(ROOT/record['artifacts']['source'])
        record['candidate_sha256'] = sha256(ROOT/record['artifacts']['candidate'])
        candidate = cv2.imread(str(ROOT/record['artifacts']['candidate']))
        record['identity'] = identity.validate(source,candidate)
        record['state'] = state.validate(source,candidate)
        record['state']['status'] = state_measurement_status(record['state'])
        record['state']['reference_frame'] = 'preprocessed_crop_not_original_full_image'
        record['status'] = ('candidate_measured' if record['identity']['status']=='measured_uncalibrated'
                            and record['state']['status']=='measured_uncalibrated' else 'validation_indeterminate')
    except Exception as error:
        record.update(status='failed',failure=str(error))
    record['elapsed_seconds'] = time.time()-started
    return record


def main(prepared, name):
    torch.set_num_threads(4);torch.manual_seed(42)
    manifest_path = ROOT/'artifacts'/prepared/'manifest.json'
    retina = json.loads(manifest_path.read_text())
    samples = json.loads((ROOT/'data/samples.json').read_text())
    prepared_rows = {row['id']:row for row in retina['samples']}
    if set(prepared_rows) != {s['id'] for s in samples}:
        raise ValueError('Prepared sample set differs from experiment sample set')
    output = new_run_directory(ROOT/'artifacts',name)
    generator = GANonymization();detector = Detector()
    identity = SFaceValidator(detector);state = StateValidator()
    experiment = {'schema_version':1,'arms':ARMS,'seed':42,'release_allowed':False,
                  'prepared_manifest_sha256':sha256(manifest_path),
                  'sample_manifest_sha256':sha256(ROOT/'data/samples.json'),
                  'generator_checkpoint_sha256':generator.manifest['sha256'],
                  'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  'code_sha256':{p.name:sha256(p) for p in [Path(__file__),ROOT/'spike.py',ROOT/'experiment_utils.py']},
                  'recognition_model_sha256':sha256(ROOT/'models/sface.onnx'),
                  'retinaface':{k:v for k,v in retina.items() if k!='samples'},
                  'generation_environment':{'torch':torch.__version__,'device':'cpu','threads':4},
                  'state_pose_scope':'Input crop vs generated crop, after any preprocessing alignment; no inverse warp',
                  'arm_difference':'YuNet uses .6 score, 1/.5 scales, 20% margin and 32px size gate; RetinaFace uses official .9 threshold, upscaling, zero margin, official crop/alignment with no additional size gate',
                  'samples':[]}
    try:
        for sample in samples:
            path = ROOT/sample['path']
            if sha256(path)!=sample['sha256'] or prepared_rows[sample['id']]['input_sha256']!=sample['sha256']:
                raise ValueError('Input differs between experimental arms: '+sample['id'])
            image = cv2.imread(str(path))
            if image is None:raise ValueError('Input decode failed')
            row = {'id':sample['id'],'input_path':sample['path'],'input_sha256':sample['sha256'],
                   'source_group':sample['source_group'],'transformed':sample['transformed'],
                   'release_allowed':False,'arms':{a:[] for a in ARMS}}
            for arm in ARMS:
                (output/arm).mkdir(exist_ok=True)
            try:
                for index,face in enumerate(detector.detect(image)):
                    x,y,w,h=map(float,face[:4]);record={'index':index,'bbox_xyxy':[x,y,x+w,y+h], 'confidence':float(face[-1])}
                    row['arms']['yunet_context'].append(record)
                    if w<32 or h<32:
                        record.update(status='failed',failure='source_face_too_small',release_allowed=False);continue
                    box=[max(0,int(x-.2*w)),max(0,int(y-.2*h)),min(image.shape[1],int(math.ceil(x+1.2*w))),min(image.shape[0],int(math.ceil(y+1.2*h)))]
                    x1,y1,x2,y2=box;record['crop_xyxy']=box
                    evaluate_face(generator,identity,state,image[y1:y2,x1:x2],output/'yunet_context',sample['id']+'-f%d'%index,record)
                    previous=ROOT/('artifacts/run-context/'+sample['id']+'-f%d-candidate.png'%index)
                    if previous.exists() and record.get('candidate_sha256'):
                        record['initial_baseline_candidate_sha256']=sha256(previous)
                        record['initial_baseline_pixel_artifact_identical']=record['candidate_sha256']==sha256(previous)
            except Exception as error:
                row['yunet_error']=str(error)
            prepared_row=prepared_rows[sample['id']]
            row['retina_preparation_status']=prepared_row['status']
            for index,face in enumerate(prepared_row['faces']):
                for arm in ARMS[1:]:
                    record={'index':index,'detector_key':face['key'],'bbox_xyxy':face['bbox_xyxy'],'confidence':face['confidence']}
                    row['arms'][arm].append(record);item=face['variants'][arm]
                    if item['status']!='prepared':
                        record.update(status='failed',failure=item.get('error','preprocessing_failed'),release_allowed=False);continue
                    crop_path=ROOT/item['path']
                    if sha256(crop_path)!=item['sha256']:
                        raise ValueError('Prepared crop checksum mismatch')
                    record['prepared_crop_sha256']=item['sha256']
                    evaluate_face(generator,identity,state,cv2.imread(str(crop_path)),output/arm,sample['id']+'-f%d'%index,record)
            row['spatial_pairs_yunet_to_retina']=match_boxes(
                [f['bbox_xyxy'] for f in row['arms']['yunet_context']],
                [f['bbox_xyxy'] for f in row['arms']['retina_unaligned']])
            for arm in ARMS:
                # These samples all depict faces, including known derived stress cases.
                row.setdefault('arm_status',{})[arm]=('zero_detections_known_face_no_release' if not row['arms'][arm]
                    else 'research_only_not_approved')
            experiment['samples'].append(row);write_json(output/'experiment.json',experiment)
            print(row['id'], {a:sum('candidate_sha256' in f for f in row['arms'][a]) for a in ARMS},flush=True)
        for arm in ARMS:
            contact_sheet([{'id':r['id'],'faces':r['arms'][arm]} for r in experiment['samples']],output/arm)
    finally:
        state.mesh.close()
    print(output/'experiment.json',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepared',required=True);parser.add_argument('--name',required=True)
    args=parser.parse_args();main(args.prepared,args.name)

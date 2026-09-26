"""Run pinned official FaceCrop in an isolated TensorFlow environment, offline.

    phase0/.retina-venv/bin/python phase0/retina_preprocess.py --name retina-v1

Detect once per image. Feed the identical detection dictionary to the unmodified
official FaceCrop for align=False and align=True. Only the detect_faces call is
temporarily replaced by its cached result; crop/alignment/color code is upstream.
No training, service worker, upload or generation runs in this process.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path
from unittest.mock import patch

from experiment_utils import new_run_directory, sha256, write_json

ROOT = Path(__file__).resolve().parent
os.environ['DEEPFACE_HOME'] = str(ROOT / 'models/retinaface')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import cv2
import numpy as np
import tensorflow as tf
import gdown
from retinaface import RetinaFace


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main(name):
    if importlib.metadata.version('retina-face') != '0.0.13':
        raise RuntimeError('Expected retina-face 0.0.13')
    weight = ROOT / 'models/retinaface/.deepface/weights/retinaface.h5'
    if not weight.is_file():
        raise FileNotFoundError('Download the official RetinaFace weights before running')
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    # Never let upstream download a missing asset during the experiment.
    gdown.download = lambda *a, **k: (_ for _ in ()).throw(RuntimeError('Runtime download disabled'))
    upstream = ROOT / 'vendor/GANonymization-main/lib/transform/face_crop_transformer.py'
    spec = importlib.util.spec_from_file_location('official_face_crop', upstream)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = new_run_directory(ROOT/'artifacts', name)
    model = RetinaFace.build_model()
    samples = json.loads((ROOT/'data/samples.json').read_text())
    manifest = {'environment': {'python':sys.version,'platform':platform.platform(),
                 'tensorflow':tf.__version__, 'retina-face':'0.0.13','numpy':np.__version__,
                 'opencv':cv2.__version__, 'device':'CPU','intra_threads':4,'inter_threads':1},
                'weight': {'source':'https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5',
                           'sha256':sha256(weight),'bytes':weight.stat().st_size},
                'official_facecrop_sha256':sha256(upstream),
                'retinaface_source_sha256':sha256(RetinaFace.__file__),
                'threshold':.9,'allow_upscaling':True,'margin':0,
                'cached_identical_detections_for_alignment_pair':True,
                'release_allowed':False,'samples':[]}
    for sample in samples:
        started = time.time()
        row = {'id':sample['id'], 'input_sha256':sample['sha256'], 'faces':[]}
        try:
            path = ROOT/sample['path']
            if sha256(path) != sample['sha256']:
                raise ValueError('Input checksum mismatch')
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError('Input decode failed')
            detections = RetinaFace.detect_faces(image, threshold=.9, model=model, allow_upscaling=True)
            row['detections'] = json.loads(json.dumps(detections, default=json_default)) if isinstance(detections,dict) else {}
            if not isinstance(detections,dict) or not detections:
                row['status'] = 'zero_detections_no_release'
            else:
                # Each subset retains the official detector key; ordering cannot silently remap faces.
                for key, detection in detections.items():
                    face = {'key':key,'bbox_xyxy':list(map(int,detection['facial_area'])),
                            'confidence':float(detection['score']), 'variants':{}}
                    row['faces'].append(face)
                    for align in (False, True):
                        arm = 'retina_aligned' if align else 'retina_unaligned'
                        item = {'status':'failed'}; face['variants'][arm]=item
                        try:
                            with patch.object(RetinaFace, 'detect_faces', return_value={key:detection}):
                                crops = module.FaceCrop(align)(image)
                            if len(crops)!=1 or crops[0].size==0:
                                raise ValueError('Official crop returned no image')
                            if not align:
                                x1,y1,x2,y2=face['bbox_xyxy']
                                if not np.array_equal(crops[0],image[y1:y2,x1:x2]):
                                    raise ValueError('Unaligned crop/color parity mismatch')
                            target = output/(sample['id']+'-'+key+'-'+arm+'.png')
                            if not cv2.imwrite(str(target),crops[0]):
                                raise IOError('Crop write failed')
                            item.update(status='prepared',path=str(target.relative_to(ROOT)),sha256=sha256(target),
                                        shape=list(crops[0].shape),color='BGR decoded via OpenCV')
                        except Exception as error:
                            item['error'] = str(error)
                row['status']='prepared'
        except Exception as error:
            row.update(status='failed',error=str(error))
        row['elapsed_seconds']=time.time()-started
        manifest['samples'].append(row)
        write_json(output/'manifest.json',manifest)
        print(row['id'], row['status'],len(row['faces']),'%.2fs'%row['elapsed_seconds'],flush=True)
    print(output/'manifest.json',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True)
    main(parser.parse_args().name)

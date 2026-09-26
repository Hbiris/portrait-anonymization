"""Local, CPU-only GANonymization feasibility measurements. Never a release API.

Uses the unmodified official GeneratorUNet and weights; does not load optimizer
objects or execute checkpoint pickle code. YuNet replaces RetinaFace cropping
(align=False). Official padding and 478-point rendering are retained verbatim.
All candidate images are unapproved research artifacts, not safe-to-upload output.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import pathlib
import platform
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.mpl'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))
sys.path.insert(0, str(ROOT/'vendor/GANonymization-main'))
import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from lib.models.pix2pix import GeneratorUNet
from lib.transform.zero_padding_resize_transformer import ZeroPaddingResize
from lib.transform.facial_landmarks_478_transformer import FacialLandmarks478
from interfaces import zero_face_decision

EXPECTED_CHECKPOINT = 'eba49bd525033b55022a91d6b4398ab088b51339fd6895db7afdd648d776eec9'

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()

def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))

class GANonymization:
    def __init__(self):
        path = ROOT/'models/publication-download'
        if digest(path) != EXPECTED_CHECKPOINT:
            raise ValueError('Checkpoint checksum mismatch')
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        self.model = GeneratorUNet().cpu().eval()
        self.model.load_state_dict({k[len('generator.'):]:v for k,v in checkpoint['state_dict'].items() if k.startswith('generator.')}, strict=True)
        self.pad = ZeroPaddingResize(512)
        self.mesh_render = FacialLandmarks478()
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((512,512)), transforms.Normalize((.5,.5,.5),(.5,.5,.5))])
        self.manifest = {'source':'https://mediastore.rz.uni-augsburg.de/get/NsLjQYey65/', 'source_label':'25 epochs (publication version)',
                         'sha256':EXPECTED_CHECKPOINT, 'bytes':path.stat().st_size,
                         'checkpoint_epoch':checkpoint.get('epoch'), 'global_step':checkpoint.get('global_step'),
                         'training_lightning_version':checkpoint.get('pytorch-lightning_version'),
                         'hyperparameters':checkpoint.get('hyper_parameters'),
                         'generator_parameters':sum(p.numel() for p in self.model.parameters()),
                         'strict_generator_load':True, 'preprocessing':'YuNet multiscale bbox crop with 20% context per side, no roll alignment; official ZeroPaddingResize and FacialLandmarks478',
                         'commercial_permission':'unresolved; research baseline only'}
        del checkpoint

    def anonymize(self, face_bgr):
        padded = self.pad(face_bgr)
        condition = self.mesh_render(padded)
        if not np.any(condition):
            raise ValueError('source_landmarks_missing')
        with torch.inference_mode():
            output = self.model(self.transform(condition).unsqueeze(0))[0].cpu()
        if not torch.isfinite(output).all():
            raise ValueError('nonfinite_generator_output')
        # Same ToPILImage conversion as official CLI. Generator channels are RGB.
        candidate_rgb = np.asarray(transforms.ToPILImage()((output+1)/2))
        return padded, candidate_rgb[:,:,::-1].copy(), condition

class Detector:
    def __init__(self):
        self.net = cv2.FaceDetectorYN.create(str(ROOT/'models/yunet.onnx'), '', (320,320), .6, .3, 5000)
    def detect(self, image):
        candidates=[]
        for scale in (1.,.5):
            view=cv2.resize(image,None,fx=scale,fy=scale) if scale!=1 else image
            self.net.setInputSize((view.shape[1],view.shape[0]))
            _, faces=self.net.detect(view)
            if faces is not None:
                for face in faces:
                    face=face.copy();face[:14]/=scale;candidates.append(face)
        if not candidates:return []
        indices=cv2.dnn.NMSBoxes([f[:4].tolist() for f in candidates],[float(f[-1]) for f in candidates],.6,.3)
        return sorted([candidates[int(i)] for i in np.asarray(indices).flatten()],key=lambda f:(float(f[0]),float(f[1])))

class SFaceValidator:
    def __init__(self, detector):
        self.detector = detector
        self.net = cv2.FaceRecognizerSF.create(str(ROOT/'models/sface.onnx'), '')
    def validate(self, original, candidate):
        a,b = self.detector.detect(original), self.detector.detect(candidate)
        if len(a)!=1 or len(b)!=1:
            return {'status':'indeterminate', 'reason':'identity_face_count', 'identity_similarity':None, 'threshold':None, 'pass':None}
        fa = self.net.feature(self.net.alignCrop(original, a[0]))
        fb = self.net.feature(self.net.alignCrop(candidate, b[0]))
        score = float(self.net.match(fa,fb,cv2.FaceRecognizerSF_FR_COSINE))
        if not math.isfinite(score):
            raise ValueError('nonfinite_identity_score')
        control = float(self.net.match(fa,fa,cv2.FaceRecognizerSF_FR_COSINE))
        return {'status':'measured_uncalibrated', 'model':'OpenCV SFace 2021dec', 'identity_similarity':score,
                'self_similarity_control':control, 'threshold':None, 'pass':None}

class StateValidator:
    def __init__(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True,min_detection_confidence=.5)
    def extract(self, image):
        result = self.mesh.process(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            return None
        p = np.array([[v.x*image.shape[1],v.y*image.shape[0]] for v in result.multi_face_landmarks[0].landmark],np.float64)
        dist = lambda a,b: float(np.linalg.norm(p[a]-p[b]))
        ear = lambda a,b,c,d,e,f: (dist(b,f)+dist(c,e))/(2*max(dist(a,d),1e-6))
        values = {'ear_33_side':ear(33,160,158,133,153,144), 'ear_263_side':ear(362,385,387,263,373,380),
                  'mouth_aperture_ratio':dist(13,14)/max(dist(78,308),1e-6),
                  'brow_33_side_ratio':dist(105,159)/max(dist(33,133),1e-6),
                  'brow_263_side_ratio':dist(334,386)/max(dist(362,263),1e-6)}
        # Approximate generic face geometry; pose is an estimate, not calibrated measurement.
        model = np.array([(0,0,0),(0,-330,-65),(-225,170,-135),(225,170,-135),(-150,-150,-125),(150,-150,-125)],np.float64)
        pixels=p[[1,152,33,263,61,291]]
        h,w=image.shape[:2]; camera=np.array([[w,0,w/2],[0,w,h/2],[0,0,1]],np.float64)
        ok, rvec,tvec=cv2.solvePnP(model,pixels,camera,np.zeros((4,1)),flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None
        rvec,tvec=cv2.solvePnPRefineLM(model,pixels,camera,np.zeros((4,1)),rvec,tvec)
        rotation,_=cv2.Rodrigues(rvec)
        projected,_=cv2.projectPoints(model,rvec,tvec,camera,np.zeros((4,1)))
        values['pose_reprojection_rmse_px']=float(np.sqrt(np.mean(np.sum((projected[:,0]-pixels)**2,axis=1))))
        values['pose_reprojection_normalized']=values['pose_reprojection_rmse_px']/max(dist(33,263),1e-6)
        values['pose_valid']=bool(tvec[2,0]>0 and values['pose_reprojection_normalized']<.15)
        values['rotation_matrix']=rotation.tolist()
        values['euler_xyz_degrees']=list(map(float,cv2.RQDecomp3x3(rotation)[0]))
        return values
    def validate(self, original, candidate):
        a,b=self.extract(original),self.extract(candidate)
        if a is None or b is None:
            return {'status':'indeterminate','reason':'state_landmarks_or_pose_missing','original':a,'candidate':b,'pass':None}
        keys=['ear_33_side','ear_263_side','mouth_aperture_ratio','brow_33_side_ratio','brow_263_side_ratio']
        delta={k:abs(a[k]-b[k]) for k in keys}
        relative=np.array(a['rotation_matrix']).T@np.array(b['rotation_matrix'])
        angle=float(np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1))))
        return {'status':'measured_uncalibrated','original':a,'candidate':b,'absolute_delta':delta,
                'head_pose_difference_degrees':angle if a['pose_valid'] and b['pose_valid'] else None,
                'head_pose_status':'approximate' if a['pose_valid'] and b['pose_valid'] else 'indeterminate_reprojection',
                'pose_method':'generic_6_point_SQPNP_LM_assumed_focal_length',
                'expression_embedding':None,'blendshapes':None,'pass':None}

def second_detector(image):
    with mp.solutions.face_detection.FaceDetection(model_selection=1,min_detection_confidence=.5) as net:
        result=net.process(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        return len(result.detections or [])

def contact_sheet(rows, output):
    tiles=[]
    for row in rows:
        for face in row.get('faces',[]):
            paths=face.get('artifacts')
            if not paths:
                continue
            tile=Image.new('RGB',(600,238),'#eeeeee'); draw=ImageDraw.Draw(tile)
            for col,key in enumerate(['source','condition','candidate']):
                pic=Image.open(ROOT/paths[key]).convert('RGB');pic.thumbnail((196,196));tile.paste(pic,(col*200,25))
            sim=face.get('identity',{}).get('identity_similarity'); pose=face.get('state',{}).get('head_pose_difference_degrees')
            draw.text((5,5),'%s / face %d | source - landmarks - candidate (UNAPPROVED)'%(row['id'],face['index']),fill='black')
            draw.text((5,221),'identity=%s pose_deg=%s'%(None if sim is None else round(sim,3),None if pose is None else round(pose,2)),fill='black')
            tiles.append(tile)
    for start in range(0,len(tiles),6):
        part=tiles[start:start+6];sheet=Image.new('RGB',(1200,238*math.ceil(len(part)/2)),'white')
        for index,tile in enumerate(part):sheet.paste(tile,((index%2)*600,(index//2)*238))
        sheet.save(output/('contact-%02d.jpg'%(start//6+1)),quality=90)

def run(limit=None, output_name='run'):
    config=json.loads((ROOT/'config.json').read_text());torch.set_num_threads(4);torch.manual_seed(config['seed'])
    output=ROOT/'artifacts'/output_name;output.mkdir(parents=True,exist_ok=True)
    generator=GANonymization();detector=Detector();identity=SFaceValidator(detector);states=StateValidator()
    environment={'python':sys.version,'platform':platform.platform(),'machine':platform.machine(),'torch':torch.__version__,
                 'opencv':cv2.__version__,'mediapipe':mp.__version__,'numpy':np.__version__,'device':'cpu','threads':4,
                 'upstream_commit':json.loads((ROOT/'artifacts/upstream-commit.json').read_text())['sha'],
                 'models':{p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in (ROOT/'models').glob('*') if p.is_file()}}
    save_json(output/'environment.json',environment);save_json(output/'checkpoint.json',generator.manifest)
    samples=json.loads((ROOT/'data/samples.json').read_text());results=[]
    for sample in samples[:limit]:
        began=time.time();image=cv2.imread(str(ROOT/sample['path']));row=dict(sample);row.update(faces=[],release_allowed=False)
        try:
            if image is None:raise ValueError('input_decode_failed')
            faces=detector.detect(image);other=second_detector(image)
            row['detectors']={'yunet':len(faces),'mediapipe_blazeface':other}
            if len(faces)==0:
                row['status']=zero_face_decision([{'status':'ok','suspected_faces':len(faces)},{'status':'ok','suspected_faces':other}],config['zero_face_policy'])
            else:
                for index,face in enumerate(faces):
                    f={'index':index,'bbox_xywh':face[:4].tolist(),'detection_confidence':float(face[-1])};row['faces'].append(f)
                    try:
                        x,y,w,h=face[:4]
                        if w<32 or h<32:raise ValueError('source_face_too_small')
                        x1=max(0,int(x-.2*w));y1=max(0,int(y-.2*h));x2=min(image.shape[1],int(math.ceil(x+1.2*w)));y2=min(image.shape[0],int(math.ceil(y+1.2*h)))
                        f['crop_xyxy']=[x1,y1,x2,y2]
                        source,candidate,condition=generator.anonymize(image[y1:y2,x1:x2])
                        stem=sample['id']+'-f%d'%index
                        f['artifacts']={key:'artifacts/'+output_name+'/'+stem+'-'+key+'.png' for key in ['source','condition','candidate']}
                        for key,pic in [('source',source),('condition',condition),('candidate',candidate)]:
                            if not cv2.imwrite(str(ROOT/f['artifacts'][key]),pic):raise ValueError('artifact_write_failed')
                        # Validate the bytes actually written, not an earlier tensor.
                        candidate=cv2.imread(str(ROOT/f['artifacts']['candidate']))
                        f['identity']=identity.validate(source,candidate);f['state']=states.validate(source,candidate)
                        f['utility']={'status':'indeterminate','independent_review':'pending','downstream_consistency':'not_implemented',
                                      'synthetic_appearance_is_care_evidence':False,'pass':None}
                        f['status']='candidate_measured' if f['identity']['status']=='measured_uncalibrated' and f['state']['status']=='measured_uncalibrated' else 'validation_indeterminate'
                    except Exception as error:
                        f['status']='failed';f['failure']=str(error)
                row['status']='research_candidates_only' if all(f['status']=='candidate_measured' for f in row['faces']) else 'rejected_partial_or_unverifiable'
                # Different detector counts represent uncertainty, not proof of coverage.
                if other>len(faces):row['status']='rejected_detector_disagreement'
        except Exception as error:
            row['status']='failed';row['failure']=str(error)
        row['elapsed_seconds']=time.time()-began;results.append(row);save_json(output/'results.json',results)
        print(row['id'],row['status'],len(row['faces']),'%.2fs'%row['elapsed_seconds'],flush=True)
    states.mesh.close();contact_sheet(results,output)
    with (output/'metrics.csv').open('w',newline='') as file:
        writer=csv.writer(file);writer.writerow(['sample','face','status','identity_cosine','eye_33_delta','eye_263_delta','mouth_delta','head_pose_delta_deg','utility','release_allowed'])
        for row in results:
            for f in row['faces'] or [{}]:
                state=f.get('state',{});d=state.get('absolute_delta',{})
                writer.writerow([row['id'],f.get('index'),f.get('status',row['status']),f.get('identity',{}).get('identity_similarity'),d.get('ear_33_side'),d.get('ear_263_side'),d.get('mouth_aperture_ratio'),state.get('head_pose_difference_degrees'),'indeterminate',False])
    return results

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);parser.add_argument('--output-name',default='run');args=parser.parse_args();run(args.limit,args.output_name)

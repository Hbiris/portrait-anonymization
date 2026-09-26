"""Focused regression checks for evidence semantics, not a production test suite."""
import json
import math
import pathlib
import unittest
from interfaces import zero_face_decision
from utility_review import IndependentVisualReviewValidator

ROOT=pathlib.Path(__file__).resolve().parent

class PolicyTests(unittest.TestCase):
    def test_zero_faces_configurable(self):
        clear=[{'status':'ok','suspected_faces':0}]*2
        self.assertEqual(zero_face_decision(clear),'reject_zero_face_policy')
        self.assertEqual(zero_face_decision(clear,'continue_if_all_clear'),'continue_no_face_policy')
    def test_any_suspicion_rejects(self):
        result=[{'status':'ok','suspected_faces':0},{'status':'ok','suspected_faces':1}]
        self.assertTrue(zero_face_decision(result,'continue_if_all_clear').startswith('reject'))
    def test_errors_and_single_detector_reject(self):
        self.assertTrue(zero_face_decision([{'status':'ok','suspected_faces':0}],'continue_if_all_clear').startswith('reject'))
        self.assertTrue(zero_face_decision([{'status':'error'},{'status':'ok','suspected_faces':0}],'continue_if_all_clear').startswith('reject'))
    def test_unknown_review_never_passes(self):
        value=IndependentVisualReviewValidator().validate(None,None,{'sample_id':'missing','face_index':0})
        self.assertIsNone(value['pass'])
    def test_visual_mismatch_is_failure(self):
        value=IndependentVisualReviewValidator().validate(None,None,{'sample_id':'s01','face_index':0})
        self.assertFalse(value['pass'])

class EvidenceTests(unittest.TestCase):
    def test_actual_run_complete_and_no_release(self):
        rows=json.loads((ROOT/'artifacts/run-context/reviewed-results.json').read_text())
        self.assertEqual(len(rows),16)
        for row in rows:
            self.assertFalse(row['release_allowed'])
            for face in row['faces']:
                if 'artifacts' in face:
                    for path in face['artifacts'].values():self.assertTrue((ROOT/path).is_file())
                identity=face.get('identity',{})
                score=identity.get('identity_similarity')
                if score is not None:
                    self.assertTrue(math.isfinite(score));self.assertGreaterEqual(score,-1.00001);self.assertLessEqual(score,1.00001)
                    self.assertAlmostEqual(identity['self_similarity_control'],1,places=4)
                    self.assertIsNone(identity['pass'])
    def test_lowlight_and_rotation_are_not_independent_photos(self):
        rows=json.loads((ROOT/'data/samples.json').read_text())
        self.assertEqual(sum(r['transformed'] for r in rows),4)
        self.assertEqual(next(r for r in rows if r['id']=='s14')['transformation'],'roll_90')

if __name__=='__main__':unittest.main()

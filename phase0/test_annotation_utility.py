import copy
import unittest
from annotation_utility import evaluate_annotations, review_template


class AnnotationTests(unittest.TestCase):
    def fixture(self):
        # Synthetic unit-test data, never reported as an actual photograph review.
        r=review_template('unit-test','test',0,'source-hash','candidate-hash')
        r.update(reviewer='synthetic-test-fixture',review_type='research_observer')
        for key,state in [('eye_state','both_open'),('mouth_state','closed'),('facial_expression','neutral'),('head_pose','frontal')]:
            r['fields'][key]={side:{'state':state,'visibility':'visible'} for side in ('original','candidate')}
        return r

    def test_old_review_cannot_apply_to_new_output(self):
        value=evaluate_annotations(self.fixture(),'source-hash','new-candidate-hash')
        self.assertEqual(value['reason'],'review_image_hash_mismatch');self.assertIsNone(value['pass'])

    def test_invented_visible_state_fails(self):
        r=self.fixture();r['fields']['eye_state']['original']={'state':'uncertain','visibility':'occluded'}
        value=evaluate_annotations(r,'source-hash','candidate-hash')
        self.assertFalse(value['pass']);self.assertIn('eye_state:generated_state_without_source_evidence',value['changed'])

    def test_closed_mouth_becomes_open_fails(self):
        r=self.fixture();r['fields']['mouth_state']['candidate']['state']='open'
        self.assertFalse(evaluate_annotations(r,'source-hash','candidate-hash')['pass'])

    def test_consistent_observer_labels_are_not_clinical_acceptance(self):
        value=evaluate_annotations(self.fixture(),'source-hash','candidate-hash')
        self.assertEqual(value['reason'],'observations_consistent_downstream_not_evaluated')
        self.assertIsNone(value['pass']);self.assertFalse(value['release_allowed'])

    def test_missing_labels_stay_indeterminate(self):
        r=self.fixture();del r['fields']['head_pose']
        self.assertIsNone(evaluate_annotations(r,'source-hash','candidate-hash')['pass'])


if __name__=='__main__':unittest.main()

import tempfile
import unittest
from pathlib import Path
from experiment_utils import bbox_iou, match_boxes, new_run_directory, state_measurement_status


class ExperimentTests(unittest.TestCase):
    def test_pairing_uses_coordinates_not_detection_order(self):
        a=[[0,0,100,100],[200,0,300,100]]
        b=[[202,0,302,100],[2,0,102,100]]
        pairs=match_boxes(a,b)
        self.assertEqual([(x['a'],x['b']) for x in pairs],[(0,1),(1,0)])

    def test_ambiguous_and_unmatched_boxes_are_not_forced_pairs(self):
        box=[0,0,100,100]
        self.assertEqual(match_boxes([box],[box,box]),[])
        self.assertEqual(match_boxes([box],[[200,200,300,300]]),[])
        self.assertEqual(bbox_iou([0,0,0,0],box),0)

    def test_existing_run_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            new_run_directory(directory,'first')
            with self.assertRaises(FileExistsError):new_run_directory(directory,'first')
            with self.assertRaises(ValueError):new_run_directory(directory,'../other')

    def test_invalid_pose_is_not_a_complete_state_measurement(self):
        value={'status':'measured_uncalibrated','head_pose_difference_degrees':None,
               'absolute_delta':dict(ear_33_side=.1,ear_263_side=.1,mouth_aperture_ratio=.1)}
        self.assertEqual(state_measurement_status(value),'indeterminate')
        value['head_pose_difference_degrees']=3.
        self.assertEqual(state_measurement_status(value),'measured_uncalibrated')
        value['absolute_delta']['ear_33_side']=float('nan')
        self.assertEqual(state_measurement_status(value),'indeterminate')


if __name__=='__main__':unittest.main()

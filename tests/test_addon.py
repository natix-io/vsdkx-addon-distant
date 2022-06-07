import numpy as np
import unittest

from vsdkx.core.structs import AddonObject, Inference
from vsdkx.addon.distant.processor import DistanceChecker


class TestAddon(unittest.TestCase):
    addon_config = {
        "camera_distance_threshold": 0.1
    }

    def test_post_process(self):
        addon_processor = DistanceChecker(self.addon_config, {}, {}, {})

        frame = (np.random.rand(640, 640, 3) * 100).astype('uint8')
        inference = Inference()

        # box to be left behind
        bb_1 = np.array([120, 150, 170, 200])
        score_1 = 0.51
        # box to be included
        bb_2 = np.array([50, 60, 250, 380])
        score_2 = 0.81

        inference.boxes = [bb_1, bb_2]
        inference.scores = [score_1, score_2]

        test_object = AddonObject(frame=frame, inference=inference, shared={})
        result = addon_processor.post_process(test_object)

        self.assertEqual(len(result.inference.boxes), 1)
        self.assertTrue((result.inference.boxes[0] == bb_2).all())


if __name__ == '__main__':
    unittest.main()

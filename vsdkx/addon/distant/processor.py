from vsdkx.core.interfaces import Addon
from vsdkx.core.structs import Inference
from numpy import ndarray


class DistanceChecker(Addon):
    """
    Checks if objects are close to the camera and filters them out from
    any objects that are further away given a certain distance threshold.
    """

    def __init__(self, addon_config: dict, model_settings: dict,
                 model_config: dict, drawing_config: dict):
        super().__init__(addon_config, model_settings, model_config,
                         drawing_config)
        self._distance_threshold = addon_config.get(
            "camera_distance_threshold", 0)

    def post_process(self, frame: ndarray, inference: Inference) -> Inference:
        """
        Check if there are people on frame close to camera, and filter the
        bounding boxes and scores of those within the borders of a certain
        distance threshold.

        Args:
            frame (ndarray): the frame data
            inference (Inference): The result of the ai

        Returns:
            updated_boxes (list): List with filtered bounding boxes
            updated_scores (list): List with filtered confidence scores
        """
        updated_boxes = []
        updated_scores = []

        for box, score in zip(inference.boxes, inference.scores):
            box_height = box[3] - box[1]
            box_width = box[2] - box[0]

            # filter people boxes by threshold
            if (box_height * box_width) > \
                    (self._distance_threshold * frame.shape[1] * frame.shape[
                        0]):
                # print(self.distance_threshold * width * height)
                # print(box_height * box_width)
                updated_boxes.append(box)
                updated_scores.append(score)
        inference.boxes = updated_boxes
        inference.scores = updated_scores
        return inference

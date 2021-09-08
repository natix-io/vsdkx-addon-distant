from vsdkx.core.interfaces import Addon, AddonObject


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

    def post_process(self, addon_object: AddonObject) -> AddonObject:
        """
        Check if there are people on frame close to camera, and filter the
        bounding boxes and scores of those within the borders of a certain
        distance threshold.

        Args:
            addon_object (AddonObject): addon object containing information
            about inference, frame, other addons shared data

        Returns:
            updated_boxes (list): List with filtered bounding boxes
            updated_scores (list): List with filtered confidence scores
        """
        updated_boxes = []
        updated_scores = []

        for box, score in zip(addon_object.inference.boxes,
                              addon_object.inference.scores):
            box_height = box[3] - box[1]
            box_width = box[2] - box[0]

            # filter people boxes by threshold
            if (box_height * box_width) > \
                    (self._distance_threshold
                     * addon_object.frame.shape[1]
                     * addon_object.frame.shape[0]):

                updated_boxes.append(box)
                updated_scores.append(score)
        addon_object.inference.boxes = updated_boxes
        addon_object.inference.scores = updated_scores
        return addon_object

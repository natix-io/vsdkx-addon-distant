# Distance Checker

This addon checks if the detected objects are close to the camera and it filters out any objects that are further away given pre-defined distance threshold. Objects that are far away from the camera, are considered low confidence detections which can cause distortions to the overall reported events, therefore, are ignored. The filtering of far away objects is executed as a `post_process` step after the model inference.

### Addon Config
```yaml
camera_distance_threshold: 0.5, # Distance ratio from the camera
```

## Debug

Example of object initialization and `post_process` execution:
```python
from vsdkx.addon.distant.processor import DistanceChecker
addon_on_config = {
  'camera_distance_threshold': 0, 
  'class': 'vsdkx.addon.distant.processor.DistanceChecker'
  }

model_config = {
    'classes_len': 1, 
    'filter_class_ids': [0], 
    'input_shape': [640, 640], 
    'model_path': 'vsdkx/weights/ppl_detection_retrain_training_2.pt'
    }
    
 model_settings = {
    'conf_thresh': 0.5, 
    'device': 'cpu', 
    'iou_thresh': 0.4
    }

distance_checker = DistanceChecker(addon_on_config, model_settings, model_config)

#post_process execution 

 addon_object = AddonObject(
    frame=np.array(RGB image), #Required RGB image in numpy format
    inference=dict{
                boxes=[array([2007,  608, 3322, 2140]), array([ 348,  348, 2190, 2145])], 
                classes=[array([0], dtype=object), array([0], dtype=object)], 
                scores=[array([0.799637496471405], dtype=object), array([0.6711544394493103], dtype=object)], 
                extra={}}, 
    shared={}
    )
 addon_object = distance_checker.post_process(addon_object)
```

This step updates the `addon_object.inference.boxes` and `addon_object.inference.scores` with the filtered bounding boxes and scores. 

**Imprtant**: This looks like a potential bug/feature request, where the `addon_object.inference.classes` should also be taken into consideration, especially if we are working with an object detector that detects the classes of multiple objects. 

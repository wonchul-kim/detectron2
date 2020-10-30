import numpy as np 
import time
import os, json
import cv2
import random
import copy
import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances

### model 
models_yaml = "COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml"
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(models_yaml))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(models_yaml)

predictor = DefaultPredictor(cfg)

register_coco_instances('val2017', {}, 
        '../datasets/annotations/instances_val2017.json', 
        '../datasets/images/val2017')
dataset_dict_meta = MetadataCatalog.get('val2017')
dataset_dict = DatasetCatalog.get('val2017')

times = []
for idx, d in enumerate(random.sample(dataset_dict, 3)):
    print(d['file_name'])
    print(os.path.exists(d['file_name']))
    img = cv2.imread(d['file_name'])
    cv2.imwrite('./res/{}_raw.png'.format(idx), img[:, :, ::-1])
    v = Visualizer(img[:, :, ::-1], metadata=dataset_dict_meta, scale=0.5)
    vis = v.draw_dataset_dict(d)
    cv2.imwrite('./res/{}_ans.png'.format(idx), vis.get_image()[:, :, ::-1])

    t_start = time.time()
    outputs = predictor(img)
    t_delta = time.time() - t_start
    times.append(t_delta)
    out = v.draw_instance_predictions(outputs['instances'].to('cpu'))
    cv2.imwrite('./res/{}_pred.png'.format(idx), out.get_image()[:, :, ::-1])


mean_t_delta = np.array(times).mean()
fps = 1/mean_t_delta
print("Average(sec):{:.2f},fps:{:.2f}".format(mean_t_delta, fps))


from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultTrainer

trainer = DefaultPredictor(cfg)

evaluator = COCOEvaluator("val2017", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "val2017")
print(inference_on_dataset(trainer.model, val_loader, evaluator))
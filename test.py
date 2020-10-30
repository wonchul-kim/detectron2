'''
https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md
https://github.com/facebookresearch/detectron2/blob/master/detectron2/model_zoo/model_zoo.py
'''
import numpy as np 
import os, json
import cv2
import random

import torch, torchvision
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


### model 
predictor = []
models_yaml = ["COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml",
               "COCO-Detection/retinanet_R_50_FPN_1x.yaml",
               "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
              ]
for i in range(3):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(models_yaml[i]))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(models_yaml[i])

    predictor.append(DefaultPredictor(cfg))

img = cv2.imread('./input.jpg')

v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)

outputs = []
for i in range(3):
    outputs.append(predictor[i](img))
    out = v.draw_instance_predictions(outputs[i]['instances'].to('cpu'))
    cv2.imwrite('./res/{}.png'.format(i), out.get_image()[:, :, ::-1])


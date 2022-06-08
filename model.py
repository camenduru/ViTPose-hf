from __future__ import annotations

import os
import subprocess
import sys

if os.getenv('SYSTEM') == 'spaces':
    import mim

    mim.uninstall('mmcv-full', confirm_yes=True)
    mim.install('mmcv-full==1.5.0', is_yes=True)

    subprocess.call('pip uninstall -y opencv-python'.split())
    subprocess.call('pip uninstall -y opencv-python-headless'.split())
    subprocess.call('pip install opencv-python-headless==4.5.5.64'.split())

import huggingface_hub
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, 'ViTPose/')

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)

HF_TOKEN = os.environ['HF_TOKEN']


class DetModel:
    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)
        self.models = self._load_models()
        self.model_name = 'YOLOX-l'

    def _load_models(self) -> dict[str, nn.Module]:
        model_dict = {
            'YOLOX-tiny': {
                'config':
                'mmdet_configs/configs/yolox/yolox_tiny_8x8_300e_coco.py',
                'model':
                'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_tiny_8x8_300e_coco/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth',
            },
            'YOLOX-s': {
                'config':
                'mmdet_configs/configs/yolox/yolox_s_8x8_300e_coco.py',
                'model':
                'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_s_8x8_300e_coco/yolox_s_8x8_300e_coco_20211121_095711-4592a793.pth',
            },
            'YOLOX-l': {
                'config':
                'mmdet_configs/configs/yolox/yolox_l_8x8_300e_coco.py',
                'model':
                'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_l_8x8_300e_coco/yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
            },
            'YOLOX-x': {
                'config':
                'mmdet_configs/configs/yolox/yolox_x_8x8_300e_coco.py',
                'model':
                'https://download.openmmlab.com/mmdetection/v2.0/yolox/yolox_x_8x8_300e_coco/yolox_x_8x8_300e_coco_20211126_140254-1ef88d67.pth',
            },
        }
        models = {
            key: init_detector(dic['config'], dic['model'], device=self.device)
            for key, dic in model_dict.items()
        }
        return models

    def set_model_name(self, name: str) -> None:
        self.model_name = name

    def detect_and_visualize(
            self, image: np.ndarray,
            score_threshold: float) -> tuple[list[np.ndarray], np.ndarray]:
        out = self.detect(image)
        vis = self.visualize_detection_results(image, out, score_threshold)
        return out, vis

    def detect(self, image: np.ndarray) -> list[np.ndarray]:
        image = image[:, :, ::-1]  # RGB -> BGR
        model = self.models[self.model_name]
        out = inference_detector(model, image)
        return out

    def visualize_detection_results(
            self,
            image: np.ndarray,
            detection_results: list[np.ndarray],
            score_threshold: float = 0.3) -> np.ndarray:
        person_det = [detection_results[0]] + [np.array([]).reshape(0, 5)]

        image = image[:, :, ::-1]  # RGB -> BGR
        model = self.models[self.model_name]
        vis = model.show_result(image,
                                person_det,
                                score_thr=score_threshold,
                                bbox_color=None,
                                text_color=(200, 200, 200),
                                mask_color=None)
        return vis[:, :, ::-1]  # BGR -> RGB


class PoseModel:
    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)
        self.models = self._load_models()
        self.model_name = 'ViTPose-B (multi-task train, COCO)'

    def _load_models(self) -> dict[str, nn.Module]:
        model_dict = {
            'ViTPose-B (single-task train)': {
                'config':
                'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py',
                'model': 'models/vitpose-b.pth',
            },
            'ViTPose-L (single-task train)': {
                'config':
                'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py',
                'model': 'models/vitpose-l.pth',
            },
            'ViTPose-B (multi-task train, COCO)': {
                'config':
                'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_base_coco_256x192.py',
                'model': 'models/vitpose-b-multi-coco.pth',
            },
            'ViTPose-L (multi-task train, COCO)': {
                'config':
                'ViTPose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py',
                'model': 'models/vitpose-l-multi-coco.pth',
            },
        }
        models = dict()
        for key, dic in model_dict.items():
            ckpt_path = huggingface_hub.hf_hub_download(
                'hysts/ViTPose', dic['model'], use_auth_token=HF_TOKEN)
            model = init_pose_model(dic['config'],
                                    ckpt_path,
                                    device=self.device)
            models[key] = model
        return models

    def set_model_name(self, name: str) -> None:
        self.model_name = name

    def predict_pose_and_visualize(
        self,
        image: np.ndarray,
        det_results: list[np.ndarray],
        box_score_threshold: float,
        kpt_score_threshold: float,
        vis_dot_radius: int,
        vis_line_thickness: int,
    ) -> tuple[list[dict[str, np.ndarray]], np.ndarray]:
        out = self.predict_pose(image, det_results, box_score_threshold)
        vis = self.visualize_pose_results(image, out, kpt_score_threshold,
                                          vis_dot_radius, vis_line_thickness)
        return out, vis

    def predict_pose(
            self,
            image: np.ndarray,
            det_results: list[np.ndarray],
            box_score_threshold: float = 0.5) -> list[dict[str, np.ndarray]]:
        image = image[:, :, ::-1]  # RGB -> BGR
        model = self.models[self.model_name]
        person_results = process_mmdet_results(det_results, 1)
        out, _ = inference_top_down_pose_model(model,
                                               image,
                                               person_results=person_results,
                                               bbox_thr=box_score_threshold,
                                               format='xyxy')
        return out

    def visualize_pose_results(self,
                               image: np.ndarray,
                               pose_results: list[np.ndarray],
                               kpt_score_threshold: float = 0.3,
                               vis_dot_radius: int = 4,
                               vis_line_thickness: int = 1) -> np.ndarray:
        image = image[:, :, ::-1]  # RGB -> BGR
        model = self.models[self.model_name]
        vis = vis_pose_result(model,
                              image,
                              pose_results,
                              kpt_score_thr=kpt_score_threshold,
                              radius=vis_dot_radius,
                              thickness=vis_line_thickness)
        return vis[:, :, ::-1]  # BGR -> RGB

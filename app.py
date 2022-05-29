#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import tarfile

if os.getenv('SYSTEM') == 'spaces':
    import mim

    mim.uninstall('mmcv-full', confirm_yes=True)
    mim.install('mmcv-full==1.5.0', is_yes=True)

    subprocess.call('pip uninstall -y opencv-python'.split())
    subprocess.call('pip uninstall -y opencv-python-headless'.split())
    subprocess.call('pip install opencv-python-headless==4.5.5.64'.split())

import gradio as gr
import huggingface_hub
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, 'ViTPose/')

from mmdet.apis import inference_detector, init_detector
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)

TOKEN = os.environ['TOKEN']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--theme', type=str)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--disable-queue',
                        dest='enable_queue',
                        action='store_false')
    return parser.parse_args()


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
            ckpt_path = huggingface_hub.hf_hub_download('hysts/ViTPose',
                                                        dic['model'],
                                                        use_auth_token=TOKEN)
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


def set_example_image(example: list) -> dict:
    return gr.Image.update(value=example[0])


def extract_tar() -> None:
    if pathlib.Path('mmdet_configs/configs').exists():
        return
    with tarfile.open('mmdet_configs/configs.tar') as f:
        f.extractall('mmdet_configs')


def main():
    args = parse_args()

    extract_tar()

    det_model = DetModel(device=args.device)
    pose_model = PoseModel(device=args.device)

    css = '''
h1#title {
  text-align: center;
}
'''

    with gr.Blocks(theme=args.theme, css=css) as demo:
        gr.Markdown('''<h1 id="title">ViTPose</h1>

This is an unofficial demo for [https://github.com/ViTAE-Transformer/ViTPose](https://github.com/ViTAE-Transformer/ViTPose).'''
                    )

        with gr.Box():
            gr.Markdown('## Step 1')
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        input_image = gr.Image(label='Input Image',
                                               type='numpy')
                    with gr.Row():
                        detector_name = gr.Dropdown(list(
                            det_model.models.keys()),
                                                    value=det_model.model_name,
                                                    label='Detector')
                    with gr.Row():
                        detect_button = gr.Button(value='Detect')
                        det_preds = gr.Variable()
                with gr.Column():
                    detection_visualization = gr.Image(
                        label='Detection Result', type='numpy')
                    vis_det_score_threshold = gr.Slider(
                        0,
                        1,
                        step=0.05,
                        value=0.5,
                        label='Visualization Score Threshold')
                    redraw_det_button = gr.Button(value='Redraw')

            with gr.Row():
                paths = sorted(pathlib.Path('images').rglob('*.jpg'))
                example_images = gr.Dataset(components=[input_image],
                                            samples=[[path.as_posix()]
                                                     for path in paths])

        with gr.Box():
            gr.Markdown('## Step 2')
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        pose_model_name = gr.Dropdown(
                            list(pose_model.models.keys()),
                            value=pose_model.model_name,
                            label='Pose Model')
                    det_score_threshold = gr.Slider(
                        0,
                        1,
                        step=0.05,
                        value=0.5,
                        label='Box Score Threshold')
                    with gr.Row():
                        predict_button = gr.Button(value='Predict')
                        pose_preds = gr.Variable()
                with gr.Column():
                    pose_visualization = gr.Image(label='Result', type='numpy')
                    vis_kpt_score_threshold = gr.Slider(
                        0,
                        1,
                        step=0.05,
                        value=0.3,
                        label='Visualization Score Threshold')
                    vis_dot_radius = gr.Slider(1,
                                               10,
                                               step=1,
                                               value=4,
                                               label='Dot Radius')
                    vis_line_thickness = gr.Slider(1,
                                                   10,
                                                   step=1,
                                                   value=2,
                                                   label='Line Thickness')
                    redraw_pose_button = gr.Button(value='Redraw')

        gr.Markdown(
            '<center><img src="https://visitor-badge.glitch.me/badge?page_id=hysts.vitpose" alt="visitor badge"/></center>'
        )

        detector_name.change(fn=det_model.set_model_name,
                             inputs=[
                                 detector_name,
                             ],
                             outputs=None)
        detect_button.click(fn=det_model.detect_and_visualize,
                            inputs=[
                                input_image,
                                vis_det_score_threshold,
                            ],
                            outputs=[
                                det_preds,
                                detection_visualization,
                            ])
        redraw_det_button.click(fn=det_model.visualize_detection_results,
                                inputs=[
                                    input_image,
                                    det_preds,
                                    vis_det_score_threshold,
                                ],
                                outputs=[
                                    detection_visualization,
                                ])

        pose_model_name.change(fn=pose_model.set_model_name,
                               inputs=[
                                   pose_model_name,
                               ],
                               outputs=None)
        predict_button.click(fn=pose_model.predict_pose_and_visualize,
                             inputs=[
                                 input_image,
                                 det_preds,
                                 det_score_threshold,
                                 vis_kpt_score_threshold,
                                 vis_dot_radius,
                                 vis_line_thickness,
                             ],
                             outputs=[
                                 pose_preds,
                                 pose_visualization,
                             ])
        redraw_pose_button.click(fn=pose_model.visualize_pose_results,
                                 inputs=[
                                     input_image,
                                     pose_preds,
                                     vis_kpt_score_threshold,
                                     vis_dot_radius,
                                     vis_line_thickness,
                                 ],
                                 outputs=[
                                     pose_visualization,
                                 ])

        example_images.click(fn=set_example_image,
                             inputs=[
                                 example_images,
                             ],
                             outputs=[
                                 input_image,
                             ])

    demo.launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()

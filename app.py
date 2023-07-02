#!/usr/bin/env python

from __future__ import annotations

import pathlib
import tarfile

import gradio as gr

from model import AppDetModel, AppPoseModel

DESCRIPTION = '# [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)'


def extract_tar() -> None:
    if pathlib.Path('mmdet_configs/configs').exists():
        return
    with tarfile.open('mmdet_configs/configs.tar') as f:
        f.extractall('mmdet_configs')


extract_tar()

det_model = AppDetModel()
pose_model = AppPoseModel()

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Box():
        gr.Markdown('## Step 1')
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    input_image = gr.Image(label='Input Image', type='numpy')
                with gr.Row():
                    detector_name = gr.Dropdown(
                        label='Detector',
                        choices=list(det_model.MODEL_DICT.keys()),
                        value=det_model.model_name)
                with gr.Row():
                    detect_button = gr.Button('Detect')
                    det_preds = gr.Variable()
            with gr.Column():
                with gr.Row():
                    detection_visualization = gr.Image(
                        label='Detection Result',
                        type='numpy',
                        elem_id='det-result')
                with gr.Row():
                    vis_det_score_threshold = gr.Slider(
                        label='Visualization Score Threshold',
                        minimum=0,
                        maximum=1,
                        step=0.05,
                        value=0.5)
                with gr.Row():
                    redraw_det_button = gr.Button(value='Redraw')

        with gr.Row():
            paths = sorted(pathlib.Path('images').rglob('*.jpg'))
            example_images = gr.Examples(examples=[[path.as_posix()]
                                                   for path in paths],
                                         inputs=input_image)

    with gr.Box():
        gr.Markdown('## Step 2')
        with gr.Row():
            with gr.Column():
                with gr.Row():
                    pose_model_name = gr.Dropdown(
                        label='Pose Model',
                        choices=list(pose_model.MODEL_DICT.keys()),
                        value=pose_model.model_name)
                det_score_threshold = gr.Slider(label='Box Score Threshold',
                                                minimum=0,
                                                maximum=1,
                                                step=0.05,
                                                value=0.5)
                with gr.Row():
                    predict_button = gr.Button('Predict')
                    pose_preds = gr.Variable()
            with gr.Column():
                with gr.Row():
                    pose_visualization = gr.Image(label='Result',
                                                  type='numpy',
                                                  elem_id='pose-result')
                with gr.Row():
                    vis_kpt_score_threshold = gr.Slider(
                        label='Visualization Score Threshold',
                        minimum=0,
                        maximum=1,
                        step=0.05,
                        value=0.3)
                with gr.Row():
                    vis_dot_radius = gr.Slider(label='Dot Radius',
                                               minimum=1,
                                               maximum=10,
                                               step=1,
                                               value=4)
                with gr.Row():
                    vis_line_thickness = gr.Slider(label='Line Thickness',
                                                   minimum=1,
                                                   maximum=10,
                                                   step=1,
                                                   value=2)
                with gr.Row():
                    redraw_pose_button = gr.Button('Redraw')

    detector_name.change(fn=det_model.set_model,
                         inputs=detector_name,
                         outputs=None)
    detect_button.click(fn=det_model.run,
                        inputs=[
                            detector_name,
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
                            outputs=detection_visualization)

    pose_model_name.change(fn=pose_model.set_model,
                           inputs=pose_model_name,
                           outputs=None)
    predict_button.click(fn=pose_model.run,
                         inputs=[
                             pose_model_name,
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
                             outputs=pose_visualization)

demo.queue(max_size=10).launch()

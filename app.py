#!/usr/bin/env python

from __future__ import annotations

import argparse
import pathlib
import tarfile

import gradio as gr

from model import AppDetModel, AppPoseModel

DESCRIPTION = '''# ViTPose

This is an unofficial demo for [https://github.com/ViTAE-Transformer/ViTPose](https://github.com/ViTAE-Transformer/ViTPose).'''
FOOTER = '<img id="visitor-badge" alt="visitor badge" src="https://visitor-badge.glitch.me/badge?page_id=hysts.vitpose" />'


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

    det_model = AppDetModel(device=args.device)
    pose_model = AppPoseModel(device=args.device)

    with gr.Blocks(theme=args.theme, css='style.css') as demo:
        gr.Markdown(DESCRIPTION)

        with gr.Box():
            gr.Markdown('## Step 1')
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        input_image = gr.Image(label='Input Image',
                                               type='numpy')
                    with gr.Row():
                        detector_name = gr.Dropdown(list(
                            det_model.MODEL_DICT.keys()),
                                                    value=det_model.model_name,
                                                    label='Detector')
                    with gr.Row():
                        detect_button = gr.Button(value='Detect')
                        det_preds = gr.Variable()
                with gr.Column():
                    with gr.Row():
                        detection_visualization = gr.Image(
                            label='Detection Result',
                            type='numpy',
                            elem_id='det-result')
                    with gr.Row():
                        vis_det_score_threshold = gr.Slider(
                            0,
                            1,
                            step=0.05,
                            value=0.5,
                            label='Visualization Score Threshold')
                    with gr.Row():
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
                            list(pose_model.MODEL_DICT.keys()),
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
                    with gr.Row():
                        pose_visualization = gr.Image(label='Result',
                                                      type='numpy',
                                                      elem_id='pose-result')
                    with gr.Row():
                        vis_kpt_score_threshold = gr.Slider(
                            0,
                            1,
                            step=0.05,
                            value=0.3,
                            label='Visualization Score Threshold')
                    with gr.Row():
                        vis_dot_radius = gr.Slider(1,
                                                   10,
                                                   step=1,
                                                   value=4,
                                                   label='Dot Radius')
                    with gr.Row():
                        vis_line_thickness = gr.Slider(1,
                                                       10,
                                                       step=1,
                                                       value=2,
                                                       label='Line Thickness')
                    with gr.Row():
                        redraw_pose_button = gr.Button(value='Redraw')

        gr.Markdown(FOOTER)

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

        example_images.click(
            fn=set_example_image,
            inputs=example_images,
            outputs=input_image,
        )

    demo.launch(
        enable_queue=args.enable_queue,
        server_port=args.port,
        share=args.share,
    )


if __name__ == '__main__':
    main()

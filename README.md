# nuScenes Dino prototype

This repository provides a minimal pipeline for streaming nuScenes sensor data into a Dino-style vision-language stack. It loads multi-camera RGB frames (with optional radar and lidar) and converts them into features using a pretrained Dino encoder so you can plug them into a downstream language decoder or control policy.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

You also need the [nuScenes dataset](https://www.nuscenes.org/download) extracted locally. Update the `dataroot` in the example script to point at your copy.

## Usage

Run the example script to iterate over a few samples and encode the camera images:

```bash
python examples/run_pipeline.py
```

By default the pipeline loads all six cameras and skips lidar/radar to keep the Dino flow fast. Set `include_radar` or `include_lidar` to `True` when constructing `NuScenesPipeline` if you want those modalities alongside the RGB images.

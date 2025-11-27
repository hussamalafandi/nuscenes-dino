# nuScenes Dino prototype

This repository provides a minimal pipeline for streaming nuScenes sensor data into a Dino-style vision-language stack. It loads multi-camera RGB frames (with optional radar and lidar) and converts them into features using a pretrained Dino encoder so you can plug them into a downstream language decoder or control policy.

## Setup

```bash
# Create and activate a Conda environment (Python 3.12)
conda create -n nuscenes-dino python=3.12 -y
conda activate nuscenes-dino

# Install PyTorch with GPU acceleration (replace CUDA version if needed)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# If you prefer CPU-only builds, install the CPU wheels instead
# conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install the remaining Python dependencies (Transformers >= 4.56 is required for DINOv3)
pip install -e .

# (Optional) Log into Hugging Face to access gated DINOv3 weights
# huggingface-cli login
```

You also need the [nuScenes dataset](https://www.nuscenes.org/download) extracted locally. Update the `dataroot` in the example script to point at your copy. If you want GPU acceleration, replace the PyTorch line with the wheel URL for your CUDA version from the [official installation guide](https://pytorch.org/get-started/locally/).

## Usage

Run the example script to iterate over a few samples and encode the camera images:

```bash
python examples/run_pipeline.py
```

If you prefer an interactive walkthrough, open the accompanying notebooks with Jupyter or VS Code:

```bash
pip install jupyterlab  # if you don't already have a notebook runner
jupyter lab examples/notebooks/nuscenes_dino_usage.ipynb
```


To visualize DINOv3 patch correspondences (adapted from the official dense/sparse tutorial) use the matching notebook:

```bash
jupyter lab examples/notebooks/dino_dense_correspondence.ipynb
```

The matching notebook downloads two web-hosted images, extracts dense patch features via `DinoImageEncoder`, and displays the
highest-scoring cross-view matches with `DinoMatcher` and `plot_correspondences`. Set `device="cuda"` or `device_map="auto"`
on the encoder for fastest results on a GPU-equipped machine.

By default the pipeline loads all six cameras and skips lidar/radar to keep the Dino flow fast. Set `include_radar` or `include_lidar` to `True` when constructing `NuScenesPipeline` if you want those modalities alongside the RGB images. `DinoImageEncoder` follows the [official DINOv3 instructions](https://github.com/facebookresearch/dinov3) by loading the Hugging Face checkpoints (default: `facebook/dinov3-convnext-tiny-pretrain-lvd1689m`) and will automatically place the model on GPU when available. Pass `device_map="auto"` (default) to let Transformers shard the model, or set `device="cpu"`/`"cuda"` to force a single device.

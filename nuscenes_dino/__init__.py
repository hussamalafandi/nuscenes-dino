"""Utilities for streaming nuScenes sensor data into a Dino-style vision model."""

from nuscenes_dino.matching import Correspondence, DenseFeatureMap, DinoMatcher, plot_correspondences
from nuscenes_dino.pipeline import NuScenesPipeline
from nuscenes_dino.vlm import DinoImageEncoder

__all__ = [
    "Correspondence",
    "DenseFeatureMap",
    "DinoImageEncoder",
    "DinoMatcher",
    "NuScenesPipeline",
    "plot_correspondences",
]
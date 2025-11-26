"""Data pipeline for preparing nuScenes samples for a vision-language model."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box, LidarPointCloud, RadarPointCloud


@dataclass
class SensorBatch:
    """Container for a single nuScenes sample with selected sensor modalities."""

    token: str
    camera_images: Dict[str, Image.Image]
    radar_points: Dict[str, RadarPointCloud] | None = None
    lidar_points: LidarPointCloud | None = None
    boxes: Dict[str, List[Box]] | None = None


class NuScenesPipeline:
    """Streams nuScenes samples and materializes them into tensors for a VLM.

    The pipeline keeps the nuScenes index in memory and opens sensor files lazily
    when iterating over samples. By default it loads RGB images for the camera
    channels provided; radar and lidar data can be requested as well but are not
    necessary for an image-only Dino prototype.
    """

    def __init__(
        self,
        dataroot: str,
        version: str = "v1.0-mini",
        camera_channels: List[str] | None = None,
        include_radar: bool = False,
        include_lidar: bool = False,
    ) -> None:
        self.dataroot = Path(dataroot)
        self.version = version
        self.camera_channels = camera_channels or [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_FRONT_LEFT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]
        self.include_radar = include_radar
        self.include_lidar = include_lidar
        self.nusc = NuScenes(version=self.version, dataroot=str(self.dataroot), verbose=False)

    def stream(self) -> Iterable[SensorBatch]:
        """Yield samples from the dataset as :class:`SensorBatch` objects."""

        for sample in self.nusc.sample:
            yield self._materialize_sample(sample_token=sample["token"])

    def _materialize_sample(self, sample_token: str) -> SensorBatch:
        sample = self.nusc.get("sample", sample_token)

        camera_images: Dict[str, Image.Image] = {}
        radar_points: Dict[str, RadarPointCloud] = {}
        boxes: Dict[str, List[Box]] = {}

        for channel in self.camera_channels:
            data_token = sample["data"].get(channel)
            if not data_token:
                continue
            sample_data = self.nusc.get("sample_data", data_token)
            img_path = Path(self.dataroot, sample_data["filename"])
            with Image.open(img_path) as img:
                camera_images[channel] = img.convert("RGB")

            box_list, _ = self.nusc.get_sample_data(data_token, box_vis_level=0)
            boxes[channel] = box_list

        if self.include_radar:
            for channel in self._radar_channels:
                data_token = sample["data"].get(channel)
                if not data_token:
                    continue
                radar_data = self.nusc.get("sample_data", data_token)
                radar_path = Path(self.dataroot, radar_data["filename"])
                radar_points[channel] = RadarPointCloud.from_file(str(radar_path))

        lidar_points_obj: LidarPointCloud | None = None
        if self.include_lidar:
            lidar_token = sample["data"].get("LIDAR_TOP")
            if lidar_token:
                lidar_data = self.nusc.get("sample_data", lidar_token)
                lidar_path = Path(self.dataroot, lidar_data["filename"])
                lidar_points_obj = LidarPointCloud.from_file(str(lidar_path))

        return SensorBatch(
            token=sample_token,
            camera_images=camera_images,
            radar_points=radar_points or None,
            lidar_points=lidar_points_obj,
            boxes=boxes or None,
        )

    @property
    def _radar_channels(self) -> List[str]:
        return [
            "RADAR_FRONT",
            "RADAR_FRONT_LEFT",
            "RADAR_FRONT_RIGHT",
            "RADAR_BACK_LEFT",
            "RADAR_BACK_RIGHT",
        ]

"""Example entrypoint for streaming nuScenes images into Dino features."""

from pathlib import Path

from tqdm import tqdm

from nuscenes_dino import DinoImageEncoder, NuScenesPipeline


def main() -> None:
    dataroot = Path("/data/nuscenes")
    pipeline = NuScenesPipeline(dataroot=str(dataroot), version="v1.0-mini", include_radar=False, include_lidar=False)
    encoder = DinoImageEncoder()

    for batch in tqdm(pipeline.stream(), desc="encoding nuScenes camera sweeps"):
        features = encoder.encode(batch.camera_images.values())
        # This is where you would pass `features` to your language decoder or downstream task.
        # For demonstration we simply break after the first batch to avoid long runtimes.
        print(f"Processed sample {batch.token} with {len(batch.camera_images)} cameras")
        break


if __name__ == "__main__":
    main()

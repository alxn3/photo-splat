import os
from pathlib import Path

import sys

sys.path.append(os.path.join(Path(__file__).resolve().parent, "NoPoSplat"))

import hydra
import torch
import numpy as np
from colorama import Fore
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from dacite import Config, from_dict
from PIL import Image

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from NoPoSplat.src.global_cfg import set_cfg
    from NoPoSplat.src.loss import get_losses
    from NoPoSplat.src.misc.LocalLogger import LocalLogger
    from NoPoSplat.src.model.decoder import get_decoder
    from NoPoSplat.src.model.encoder import get_encoder
    from NoPoSplat.src.model.model_wrapper import ModelWrapper
    from NoPoSplat.src.config import DecoderCfg, EncoderCfg
    from NoPoSplat.src.dataset.types import BatchedExample, BatchedViews
    from NoPoSplat.src.model.ply_export import export_ply


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


def load_pair(pair: tuple[Path, Path]) -> torch.Tensor:
    """
    Load a pair of images from the given paths and return them as a tensor.
    """
    image1 = np.array(Image.open(pair[0]))
    image2 = np.array(Image.open(pair[1]))

    # Convert to float32 and normalize to [0, 1]
    image1 = image1.astype(np.float32) / 255.0
    image2 = image2.astype(np.float32) / 255.0

    # Stack the images along the batch dimension
    images = np.stack([image1, image2], axis=0)

    # Remove the alpha channel if it exists
    if images.shape[-1] == 4:
        images = images[..., :3]

    return torch.from_numpy(images).permute(0, 3, 1, 2).float().cuda()


def predict(model: ModelWrapper, batch):
    batch = model.data_shim(batch)

    visualization_dump = {}
    with model.benchmarker.time("encoder"):
        encoder_output = model.encoder(
            batch["context"], model.global_step, visualization_dump
        )
    return encoder_output, visualization_dump


def run(
    model: ModelWrapper,
    input: list[tuple[Path, Path]],
    output_dir: Path,
    resolution: tuple[int, int],
):
    """
    Perform inference on the given batch using the model and save the output to the specified directory.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for i, pair in enumerate(input):
        print(cyan(f"Processing pair {i + 1}/{len(input)}: {pair}"))

        # Load the images from the batch
        images = load_pair(pair)

        # Resize the images to the specified resolution
        images: torch.Tensor = torch.nn.functional.interpolate(
            images, size=resolution, mode="bilinear", align_corners=False
        )
        images = images.unsqueeze(0)

        # Create a batch with the loaded images
        batch = BatchedExample(
            context=BatchedViews(
                intrinsics=torch.tensor(
                    [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [0.0, 0.0, 1.0]]
                )
                .float()
                .reshape(1, 1, 3, 3)
                .repeat(1, 2, 1, 1)
                .cuda(),
                image=images,
            )
        )

        # Perform inference
        with torch.no_grad() and torch.autocast(device_type="cuda"):
            gaussian, visualization_dump = predict(model, batch)

        # Save the output to the specified directory
        for b in range(gaussian.means.shape[0]):
            means = gaussian.means[b]
            # covariances = gaussian.covariances[b]
            harmonics = gaussian.harmonics[b]
            opacities = gaussian.opacities[b]
            rotations = visualization_dump["rotations"][b]
            scales = visualization_dump["scales"][b]
            export_ply(
                means=means,
                scales=scales,
                rotations=rotations,
                harmonics=harmonics,
                opacities=opacities,
                path=output_dir / f"output_{b}.ply",
            )


@hydra.main(
    version_base=None,
    config_path="./NoPoSplat/config",
    config_name="predict",
)
def train(cfg_dict: DictConfig):
    set_cfg(cfg_dict)

    inputs: list[tuple[Path, Path]] = []
    # Check if input path contains two images or folders
    if not os.path.exists(cfg_dict.input):
        raise FileNotFoundError(f"Input path {cfg_dict.input} does not exist.")
    elif os.path.isfile(cfg_dict.input):
        raise ValueError(
            f"Input path {cfg_dict.input} must be a directory containing images or folders containing two images."
        )
    elif os.path.isdir(cfg_dict.input) and len(os.listdir(cfg_dict.input)) == 0:
        raise ValueError(f"Input directory {cfg_dict.input} is empty.")
    elif (
        os.path.isdir(cfg_dict.input)
        and len(os.listdir(cfg_dict.input)) == 1
        and os.path.isfile(os.path.join(cfg_dict.input, os.listdir(cfg_dict.input)[0]))
    ):
        raise ValueError(
            f"Input directory {cfg_dict.input} must contain at least 2 images."
        )

    num_files = len(os.listdir(cfg_dict.input))
    if num_files == 2 and all(
        os.path.isfile(os.path.join(cfg_dict.input, f))
        for f in os.listdir(cfg_dict.input)
    ):
        inputs = [
            tuple(os.path.join(cfg_dict.input, f) for f in os.listdir(cfg_dict.input))
        ]
    else:
        if directories := [
            os.path.join(cfg_dict.input, d) for d in os.listdir(cfg_dict.input)
        ]:
            for directory in directories:
                if os.path.isdir(directory):
                    images = [
                        os.path.join(directory, f)
                        for f in os.listdir(directory)
                        if os.path.isfile(os.path.join(directory, f))
                    ]
                    if len(images) == 2:
                        inputs.append(tuple(images))
                    else:
                        raise ValueError(
                            f"Input directory {directory} must contain exactly 2 images."
                        )
        else:
            raise ValueError(
                f"Input directory {cfg_dict.input} must contain at least 2 images or folders containing 2 images."
            )
    # Check if the images are in the correct format
    for pair in inputs:
        for image in pair:
            if not os.path.isfile(image):
                raise ValueError(f"Input {image} is not a valid file.")
            if not image.lower().endswith((".png", ".jpg", ".jpeg")):
                raise ValueError(f"Input {image} is not a valid image format.")

    # Set up the output directory.
    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    print(cyan(f"Saving outputs to {output_dir}."))

    # Prepare the checkpoint for loading.
    checkpoint_path: str = cfg_dict.checkpointing.load

    torch.manual_seed(cfg_dict.seed)

    encoder_cfg: EncoderCfg = from_dict(
        data_class=EncoderCfg, data=OmegaConf.to_container(cfg_dict.model.encoder)
    )

    encoder, encoder_visualizer = get_encoder(encoder_cfg)

    decoder_cfg: DecoderCfg = from_dict(
        data_class=DecoderCfg, data=OmegaConf.to_container(cfg_dict.model.decoder)
    )

    decoder = get_decoder(decoder_cfg)

    model_wrapper = ModelWrapper(
        None,
        None,
        None,
        encoder,
        encoder_visualizer,
        decoder,
        None,
        None,
        None,
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    model_wrapper.load_state_dict(state_dict)
    model_wrapper.cuda()

    run(model_wrapper, inputs, output_dir, cfg_dict.resolution)


if __name__ == "__main__":
    train()

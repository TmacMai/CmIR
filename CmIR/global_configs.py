"""Global dataset feature dimensions and compute device."""

import torch

TEXT_DIM = 0
ACOUSTIC_DIM = 0
VISUAL_DIM = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Unimodal feature dimensions of the pre-extracted .pkl datasets.
# Note: CMU-MOSEI visual features have 35 dimensions in the standard pickle
# release (vs. 47 for CMU-MOSI).
_DATASET_CONFIGS = {
    "mosi": {"ACOUSTIC_DIM": 74, "VISUAL_DIM": 47, "TEXT_DIM": 768},
    "mosei": {"ACOUSTIC_DIM": 74, "VISUAL_DIM": 35, "TEXT_DIM": 768},
    "sims": {"ACOUSTIC_DIM": 33, "VISUAL_DIM": 709, "TEXT_DIM": 768},
}


def set_dataset_config(dataset_name):
    global TEXT_DIM, ACOUSTIC_DIM, VISUAL_DIM

    config = _DATASET_CONFIGS.get(dataset_name)
    if config is None:
        raise ValueError(f"Invalid dataset name: {dataset_name}")
    ACOUSTIC_DIM = config["ACOUSTIC_DIM"]
    VISUAL_DIM = config["VISUAL_DIM"]
    TEXT_DIM = config["TEXT_DIM"]

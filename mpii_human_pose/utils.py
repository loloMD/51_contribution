from pathlib import Path
import scipy.io as sio
import numpy as np
from rich import print
import time

def load_matfile_anns(matfile_path: Path) -> dict:
    """Load the .mat file containing annotations (should be called `mpii_human_pose_v1_u12_1.mat`)

    This .mat annotation file requires scipy package to be loaded (-> MATLAB 5.0 FILE). TODO: Add smaller package to load the .mat file

    Args:
        matfile_path (Path): Path of the .mat file containing annotations (should be called `mpii_human_pose_v1_u12_1.mat`)

    Returns:
        np.ndarray: np array containing the annotations
    """
    print(f"Loading annotations file from {matfile_path}")
    time_it = time.time()
    annotations_file: np.ndarray = sio.loadmat(matfile_path, simplify_cells=True)["RELEASE"]
    print(f"Loaded annotations file in {time.time()-time_it} seconds")

    if tuple(annotations_file.keys()) != (
        "annolist",
        "img_train",
        "version",
        "single_person",
        "act",
        "video_list",
    ):
        raise ValueError(
            "The .mat file is not in the expected format. Please check the file and try again OR this code is no longer available"
        )

    return annotations_file


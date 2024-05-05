from pathlib import Path
import scipy.io as sio
import numpy as np
from tqdm import tqdm
from rich import print

def not_contain_nested_obj(arr) -> bool:
    """Check if a numpy array is not nested

    Args:
        arr (_type_): np array

    Returns:
        bool: True if the array is not nested, False otherwise
    """
    return not arr.dtype.hasobject


def np_arr_to_dict(arr) -> dict:
    """convert a numpy array with nested 'O' dtype items.

    Args:
        arr (_type_): np array

    Example :
     array_to_process = array([[(array([[3.88073395]]), array([[(array([[601]], dtype=uint16), array([[380]], dtype=uint16))]],
              dtype=[('x', 'O'), ('y', 'O')]))                                                         ]],
      dtype=[('scale', 'O'), ('objpos', 'O')])

      # should output
      {
        'scale': 3.88073395,
        'objpos': {
            'x': 601,
            'y': 380
        }
      }

    Returns:
        dict: dict result of the conversion
    """
    import pdb;pdb.set_trace()

    
    if not_contain_nested_obj(arr):
        return arr.item() if arr.size == 1 else None

    # exploding array in dict of str : np_array
    res = []
    for row in arr:
        res.append(
            dict(zip(arr.dtype.names, row))
        )

    # recursively converting the nested arrays
    for i, row in tqdm(enumerate(res), desc="Processing nested arrays"):
        for key, value in row.items():
            res[i][key] = np_arr_to_dict(value)

    return res


def load_matfile_anns(matfile_path: Path) -> dict:
    """Load the .mat file containing annotations (should be called `mpii_human_pose_v1_u12_1.mat`)

    This .mat annotation file requires scipy package to be loaded (-> MATLAB 5.0 FILE). TODO: Add smaller package to load the .mat file

    Args:
        matfile_path (Path): Path of the .mat file containing annotations (should be called `mpii_human_pose_v1_u12_1.mat`)

    Returns:
        np.ndarray: np array containing the annotations
    """
    annotations_file: np.ndarray = sio.loadmat(matfile_path, simplify_cells=True)["RELEASE"]

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


if __name__ == "__main__":
    import numpy as np

    root_path = Path(
        "/home/lolo/Documents/51_contribution/mpii_human_pose/mpii_human_pose_data"
    )

    path_anns_matfile = (
        root_path / "mpii_human_pose_v1_u12_2" / "mpii_human_pose_v1_u12_1.mat"
    )
    path_img_2_vid_matfile = root_path / "mpii_human_pose_v1_sequences_keyframes.mat"
    path_imgs = root_path / "images"

    annotations_file: np.ndarray = load_matfile_anns(path_anns_matfile)

    # ((1, 24987), (1, 24987), (1,), (24987, 1), (24987, 1), (1, 2821)
    annolist, img_train, _, single_person, act, video_list = annotations_file.item()

    import pdb; pdb.set_trace()

    print(np_arr_to_dict(video_list.flatten()))

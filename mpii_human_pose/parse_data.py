from typing import List, Literal, Tuple, Union
import fiftyone as fo
import fiftyone.core.utils as focu
import eta.core.utils as etau
from pathlib import Path
import numpy as np
from rich import print
from PIL import Image

assert focu.ensure_import(
    "scipy", error_msg="Please install the `scipy` package to load the .mat file"
)

from utils import load_matfile_anns
from utils_typing import (
    File_Annotation,
    Activity_annotation,
    Img_annotation,
    Body_annotation,
    Keypoint_annotation,
    Keypoint,
    Img_name,
    Point,
    JOINT_ID,
)

# This will populate the description of the fo.Dataset fields  
FIELD_DESCRIPTIONS = {
    "video_id": "specifies video id as is provided by YouTube. To watch video on youtube go to https://www.youtube.com/watch?v=video_list(videoidx)",
    "frame_sec": "image position in video, in seconds",
    "rectangle_id": "rectangle id ridx of sufficiently separated individuals",
    "activity": "activity/category name & activity id",
    "head_rect": "coordinates of the head rectangle (one row per person)",
    "objpos": "rough human position in the image (one row per person)",
    "scale": "person scale w.r.t. 200 px height (one row per person)",
    "annopoints": "51 Keypoints",
}


def safe_empty_array_checking(entity: any) -> any:
    """If an entity contain an empty array, return None
        otherwise return the entity
    Args:
        entity (any): any entity (list, np.array, dict, ...)

    Returns:
        None if in the form of an empty np.array, otherwise the entity
    """
    if type(entity) == np.ndarray:
        if entity.size == 0:
            return None
    return entity


def process_body_annotations(
    body_annotations: List[Body_annotation],
    img_width: int,
    img_height: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    fo.Keypoints,
]:
    """
    Keys in Body_annotation dict are all optional ! and transform it into 51 Keypoint

    Returns:
        dict: a user-friendly dict equivalent of the body_annotations parameter
    """

    coord_head_rect: List[List[int]] = []
    objpos: List[List[int]] = []
    scale: List[float] = []
    annopoints: List[fo.Keypoint] = []

    # print(body_annotations)
    for body_annotation in body_annotations:
        if (
            "x1" in body_annotation
        ):  # if 'x1' in dict keys, then 'x2', 'y1', 'y2' are also in the dict
            coord_head_rect.append(
                [
                    body_annotation["x1"],
                    body_annotation["y1"],
                    body_annotation["x2"],
                    body_annotation["y2"],
                ]
            )   

        if "objpos" in body_annotation:
            objpos_item = safe_empty_array_checking(body_annotation["objpos"])
            if objpos_item is not None:
                objpos.append([objpos_item["x"], objpos_item["y"]])

        if "scale" in body_annotation:
            scale_item = safe_empty_array_checking(body_annotation["scale"])
            if scale_item is not None:
                scale.append(scale_item)

        if "annopoints" in body_annotation:
            body_joint_annotations = safe_empty_array_checking(body_annotation["annopoints"])
            if body_joint_annotations is not None:
                body_joint_annotations = body_joint_annotations["point"]
                if type(body_joint_annotations) != list:
                    body_joint_annotations = [body_joint_annotations]
                annopoints.append(extract_51_keypoint(body_joint_annotations, img_width, img_height))

    return np.array(coord_head_rect), np.array(objpos), np.array(scale), fo.Keypoints(keypoints=annopoints)


def extract_51_keypoint(keypoints: List[Keypoint], img_width : int, img_height : int) -> fo.Keypoint:
    # print(f'{keypoints=}')
    points = list(map(lambda x: (x["x"]/img_width, x["y"]/img_height), keypoints))
    joints_id = list(map(lambda x: JOINT_ID[x["id"]], keypoints))
    
    if "is_visible" in keypoints[0]: # if 'is_visible' is among the dict keys of one element, then its for all elements 
        is_visible = list(map(lambda x: bool(x["is_visible"]) if type(x["is_visible"])==int else None, keypoints))
    else:
        is_visible = [None]*len(keypoints)

    return fo.Keypoint(points=points, visible=is_visible, joints_id=joints_id)


def parse_data(
    matfile_path: Path, root_imgs: Path, img_2_vid_mat_file: Path
) -> fo.Dataset:
    """Parse the .mat file found in `http://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz` Annotations folder.
    in order to create a FiftyOne dataset.

    This .mat annotation file requires scipy package to be loaded (-> MATLAB 5.0 FILE). TODO: Add smaller package to load the .mat file

    Args:
        matfile_path (Path): Path of the .mat file containing annotations (should be called `mpii_human_pose_v1_u12_1.mat`)
        root_imgs (Path): Path to the root folder containing the images (`mpii_human_pose_v1`)
        img_2_vid_mat_file (Path): Path to the .mat file containing the mapping between images and videos

    Returns:
        fo.Dataset: Fiftyone Dataset ready to be loaded in the APP
    """

    # TODO: See how to leverage this
    # img_2_vid = sio.loadmat(img_2_vid_mat_file)["annolist_keyframes"]
    # img_2_vid: dict = dict(
    #     map(lambda x: x[0].item()[0].item().split("/"), img_2_vid.flatten().tolist())
    # )

    # YES WE HAVE ANNOTATION ASSOCIATED TO IMAGE THAT DOES NOT EXIST !!!!!!!!
    ORPHAN_ANNOTATION = {}

    RAW_ANNOTATIONS_DATA: File_Annotation = load_matfile_anns(matfile_path)

    annolist = RAW_ANNOTATIONS_DATA["annolist"]
    img_train: Tuple[Literal[0, 1]] = tuple(RAW_ANNOTATIONS_DATA["img_train"].tolist())
    single_person: Tuple[Union[List[int], int]] = tuple(
        map(
            lambda x: x.tolist() if type(x) == np.ndarray else x,
            RAW_ANNOTATIONS_DATA["single_person"].tolist(),
        )
    )
    act = RAW_ANNOTATIONS_DATA["act"]
    video_list: List[str] = RAW_ANNOTATIONS_DATA["video_list"].tolist()  # len()==2821

    samples = []
    with etau.ProgressBar(total=len(annolist), start_msg=f'Parsing raw annotations to a fiftyone.Sample format') as pb:
        for annotation, is_train, rectangle_id, activity in pb(zip(
            annolist, img_train, single_person, act
        )):
            
            image_name = annotation["image"]["name"]
            if not (root_imgs / image_name).exists():
                ORPHAN_ANNOTATION[image_name] = annotation
                print(f"Skipping annotation for {image_name} ---> does not exist on the disk")
                continue

            # ------ parse Activity_annotation
            category_name = safe_empty_array_checking(activity["cat_name"])
            activity_name = safe_empty_array_checking(activity["act_name"])
            activity_id = activity["act_id"]

            ## ------ parse Img_annotation
            body_annotations = safe_empty_array_checking(annotation["annorect"])
            video_idx = safe_empty_array_checking(annotation["vididx"])
            frame_sec = safe_empty_array_checking(annotation["frame_sec"])

            ## ------ parse Body_annotation
            img_path = root_imgs / image_name

            IMAGE_WIDTH, IMAGE_HEIGHT = Image.open(img_path).size

            # print(f'{body_annotations=}')
            if body_annotations is not None:
                if type(body_annotations) != list :
                    body_annotations = [body_annotations]
                coord_head_rect, objpos, scale, annopoints = process_body_annotations(body_annotations, IMAGE_WIDTH, IMAGE_HEIGHT)

            # ---------------- CREATING 51-SAMPLE
            sample = fo.Sample(
                filepath=str(img_path), 
                tags=["train" if is_train else "test"]
            )


            sample["video_id"] = video_list[video_idx-1] if video_idx is not None else None

            sample["frame_sec"] = frame_sec

            sample["rectangle_id"] = [rectangle_id] if type(rectangle_id) == int else rectangle_id

            # TODO: use fo.Classification
            sample["activity"] = {
                "activity_name": activity_name,
                "category_name": category_name,
                "activity_id": activity_id,
            }

            # TODO: use fo.Detection : but need to convert x,y,x,y -> x,y,w,h
            # print(f'{coord_head_rect.shape=}') 
            sample["head_rect"] =  coord_head_rect if coord_head_rect.size != 0 else None

            sample["objpos"] = objpos

            sample["scale"] = scale

            # import pdb; pdb.set_trace()
            sample["annopoints"] = annopoints

            samples.append(sample)
        

    dataset = fo.Dataset("MPII Human Pose")
    dataset.add_samples(samples)
    dataset.tags = [
        "version1",
        "MPII Human Pose",
    ]

    dataset.persistent = True

    add_field_descriptions(dataset)

    return dataset, ORPHAN_ANNOTATION

def add_field_descriptions(dataset: fo.Dataset):
    global FIELD_DESCRIPTIONS
    for field, description in FIELD_DESCRIPTIONS.items():
        dataset.info[field] = description


if __name__ == "__main__":
    root_path = Path(
        "/home/lolo/Documents/51_contribution/mpii_human_pose/mpii_human_pose_data"
    )

    path_anns_matfile = (
        root_path / "mpii_human_pose_v1_u12_2" / "mpii_human_pose_v1_u12_1.mat"
    )
    path_img_2_vid_matfile = root_path / "mpii_human_pose_v1_sequences_keyframes.mat"
    path_imgs = root_path / "images"

    mpiiHP_51_dataset, orphan_anns = parse_data(path_anns_matfile, path_imgs, path_img_2_vid_matfile)
    print(len(orphan_anns))
    print(orphan_anns)
    print(mpiiHP_51_dataset)


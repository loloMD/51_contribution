"""
Detailed typing for the annotation file.

http://human-pose.mpi-inf.mpg.de/#download

Annotation description

.annolist(imgidx) - annotations for image imgidx
    .image.name - image filename
    .annorect(ridx) - body annotations for a person ridx
        .x1, .y1, .x2, .y2 - coordinates of the head rectangle
        .scale - person scale w.r.t. 200 px height
        .objpos - rough human position in the image
        .annopoints.point - person-centric body joint annotations
            .x, .y - coordinates of a joint
            id - joint id (0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist)
            is_visible - joint visibility
    .vidx - video index in video_list
    .frame_sec - image position in video, in seconds

img_train(imgidx) - training/testing image assignment (1 - train, 0 - test)

single_person(imgidx) - contains rectangle id ridx of sufficiently separated individuals

act(imgidx) - activity/category label for image imgidx
    act_name - activity name
    cat_name - category name
    act_id - activity id

video_list(videoidx) - specifies video id as is provided by YouTube. To watch video on youtube go to https://www.youtube.com/watch?v=video_list(videoidx)

"""

from typing import List, TypedDict, Required, NotRequired, Union

from enum import Enum

import numpy as np

# 0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist
JOINT_ID = (
    "r_ankle",
    "r_knee",
    "r_hip",
    "l_hip",
    "l_knee",
    "l_ankle",
    "pelvis",
    "thorax",
    "upper_neck",
    "head_top",
    "r_wrist",
    "r_elbow",
    "r_shoulder",
    "l_shoulder",
    "l_elbow",
    "l_wrist",
)


class Point(TypedDict):
    x: int
    y: int


class Img_name(TypedDict):
    name: str  # *.jpg


class Keypoint(TypedDict):
    id: int  # see JOINT_ID tuple above
    is_visible: NotRequired[
        Union[int, np.ndarray]
    ]  # O or 1, but cases `array([], dtype=uint8)` are present ...
    x: float
    y: float


class Keypoint_annotation(TypedDict):
    point: Union[List[Keypoint], Keypoint]  # !!!! NOT ALWAYS LEN 16 (even can have 1 keypoint) !!!!


class Body_annotation(TypedDict):  # body annotations for a person ridx
    x1: NotRequired[int]
    y1: NotRequired[int]
    x2: NotRequired[int]
    y2: NotRequired[int]
    scale: NotRequired[
        Union[int, float, np.ndarray]
    ]  # int or float or `array([], dtype=float64)`
    objpos: NotRequired[Union[Point, np.ndarray]]  # Point or `array([], dtype=float64)`
    annopoints: NotRequired[
        Union[Keypoint_annotation, np.ndarray]
    ]  # Keypoint_annotation or `array([], dtype=uint8)`

    # TODO : DEMISTIFY MEANINGS OF THOSE 3 GROUPS OF ATTRIBUTES BELOW

    head_r11: float  # seems to be float between -1 and 1
    head_r12: float
    head_r13: float
    head_r21: float
    head_r22: float
    head_r23: float
    head_r31: float
    head_r32: float
    head_r33: float

    torso_r11: float  # same as above
    torso_r12: float
    torso_r13: float
    torso_r21: float
    torso_r22: float
    torso_r23: float
    torso_r31: float
    torso_r32: float
    torso_r33: float

    part_occ1: int  # seems to be 0 or 1
    part_occ2: int
    part_occ3: int
    part_occ4: int
    part_occ5: int
    part_occ6: int
    part_occ7: int
    part_occ8: int
    part_occ9: int
    part_occ10: int


class Img_annotation(TypedDict):
    image: Img_name
    annorect: Union[
        List[Body_annotation], Body_annotation, np.ndarray
    ]  # if np.ndarray ---> `array([], dtype=uint8)`
    vididx: Union[int, np.ndarray]  # int 1-2821 OR `array([], dtype=float64)` (!!!! STARTING FROM 1 !!!!!!)
    frame_sec: Union[float, np.ndarray]  # int OR `array([], dtype=float64)`


class Activity_annotation(TypedDict):
    act_name: Union[
        str, np.ndarray
    ]  # ''horse racing', 'driving automobile or light truck', ... OR empty `array([], dtype='<U1')`
    cat_name: Union[
        str, np.ndarray
    ]  # 'dancing', 'sports', ... OR empty `array([], dtype='<U1')`
    act_id: int  # 630 different ids !!! non-contiguous !!! (-1 mean no activity)


class File_Annotation(TypedDict):
    annolist: List[Img_annotation]  # len: 24987
    img_train: np.ndarray  # (24987, ) only 0 or 1
    version: str
    single_person: (
        np.ndarray
    )  # (24987, ) dtype: object --> Union[int, np.ndarray 1D of contiguous int starting at 1]
    act: List[Activity_annotation]  # len: 24987
    video_list: int

import fiftyone as fo
import fiftyone.core.utils as focu
import eta.core.utils as etau
from pathlib import Path
import numpy as np
from rich import print

assert focu.ensure_import('scipy', error_msg="Please install the `scipy` package to load the .mat file")
import scipy.io as sio

def process_body_annotations(body_annotations : np.ndarray) -> dict:
    """Process the body annotations of a person in the image.

    .annorect(ridx) - body annotations for a person ridx
            .x1, .y1, .x2, .y2 - coordinates of the head rectangle
            .scale - person scale w.r.t. 200 px height
            .objpos - rough human position in the image
            .annopoints.point - person-centric body joint annotations
                .x, .y - coordinates of a joint
                id - joint id (0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist)
                is_visible - joint visibility

    Args:
        body_annotations (np.ndarray): VERY UNFRIENDLY STRUCTURE OF nested np arrays with objects dtypes

    Returns:
        dict: a user-friendly dict equivalent of the body_annotations parameter
    """

    #TODO : Implement this function

    import pdb; pdb.set_trace()
    res = dict(zip(body_annotations.dtype.names, body_annotations.item()))

    if 'scale' in res:
        res['scale'] = res['scale'].item()

    if 'objpos' in res:
        res['objpos'] = dict(zip( res['objpos'].dtype.names, tuple(map(lambda x : x.item(), res['objpos'].item()))))

    return res


def parse_data(matfile_path : Path, root_imgs : Path, img_2_vid_mat_file : Path) -> fo.Dataset:
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

    annotations_file : np.ndarray = sio.loadmat(matfile_path)['RELEASE']

    img_2_vid = sio.loadmat(img_2_vid_mat_file)['annolist_keyframes']
    img_2_vid : dict =  dict(map( lambda x : x[0].item()[0].item().split('/') , img_2_vid.flatten().tolist()))

    if annotations_file.dtype.names != ('annolist', 'img_train', 'version', 'single_person', 'act', 'video_list'):
        raise ValueError("The .mat file is not in the expected format. Please check the file and try again OR this code is no longer available")

    '''
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
    '''
    annolist, img_train, _, single_person, act, video_list = annotations_file.item() # np.ndarray  ((1, 24987), (1, 24987), (1,), (24987, 1), (24987, 1), (1, 2821))

    video_list : tuple = tuple(map(lambda x : x.item(), video_list.flatten().tolist())) 

    samples = []
    # with etau.ProgressBar() as pb:
    for annotation, is_train, rectangle_id, activity in zip(annolist.flatten(), img_train.flatten(), single_person.flatten(), act.flatten()) : #pb(zip(annolist.flatten(), img_train.flatten(), single_person.flatten(), act.flatten())):

        # parse annotations info
        image_name, body_annotations, video_idx, frame_sec =  annotation
        
        image_name =  image_name.item()[0].item()
        body_annotations = process_body_annotations(body_annotations)
        
        video_idx = None if video_idx.size == 0 else video_idx.item()
        video_idx = video_list[video_idx] if video_idx is not None else None
        
        frame_sec = None if frame_sec.size == 0 else frame_sec.item()

        # parse activity label info
        activity_name, category_name, activity_id =  tuple(map(lambda x : x.item() if x.size!=0 else None , activity))
        
        print(rectangle_id)
        rectangle_id = None if rectangle_id.size == 0 else rectangle_id.flatten().tolist()

        img_path = root_imgs / image_name
        sample = fo.Sample(filepath=str(img_path), tags=["train" if is_train else "test"])

        sample['yt_video'] = {
            'video_id': video_idx,
            'frame_sec': frame_sec
        }

        sample['rectangle_id'] = rectangle_id

        sample['activity'] = {
            'activity_name': activity_name,
            'category_name': category_name,
            'activity_id': activity_id
        }

        sample['body_annotations'] = body_annotations

        if is_train:
            import pdb; pdb.set_trace()
        import pdb; pdb.set_trace()

        samples.append(sample)

    
    dataset = fo.Dataset("MPII Human Pose")
    dataset.add_samples(samples)
    dataset.tags = ['version1', 'MPII Human Pose', ]
    #TODO : add classes | keypoint skeleton attributes


if __name__ == "__main__":
    root_path = Path('/home/lolo/Documents/51_contribution/mpii_human_pose/mpii_human_pose_data')
    
    path_anns_matfile = root_path / 'mpii_human_pose_v1_u12_2' / 'mpii_human_pose_v1_u12_1.mat'
    path_img_2_vid_matfile = root_path / 'mpii_human_pose_v1_sequences_keyframes.mat'
    path_imgs = root_path / 'images'

    parse_data(path_anns_matfile, path_imgs, path_img_2_vid_matfile)

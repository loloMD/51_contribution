# Portage of MPII Human Pose Dataset to FiftyOne format

FiftyOne dataset is ready to be downloaded from the Hugging Face Hub [here](https://github.com/loloMD/51_contribution/tree/mpii_human_pose/mpii_human_pose)

There is the code to convert the MPII Human Pose dataset to FiftyOne format.

## Raw data file needed

The raw data file is available on the MPII Human Pose dataset [website](http://human-pose.mpi-inf.mpg.de/#download).

You only need to download 2 files :

* [Images directory](https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1.tar.gz)

* [Annotations file](https://datasets.d2.mpi-inf.mpg.de/andriluka14cvpr/mpii_human_pose_v1_u12_2.zip)

## Usage

```bash
python parse_data.py
```

TODO: **Better Packaging of the code (CLI interface, Earthfile, etc.) is coming soon** !!!!!!!!!!!!!!!!!

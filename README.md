# Pericolo - Perspective correction
> **Disclaimer**
> 
> This repository is build by Walid Benbihi as part of a freelance contract at the exclusive discretion of Pericolo teams.

# Project structure
The following repository is constructed as follow
```bash
bin\                # Binary scripts for bash execution
data\               # Folder for data exploration
doc\                # Pericolo library Documentation
libs\               # Pericolo library
markers\            # Marker's images for SIFT
notebooks\          # Jupyter Notebooks
utils\              # Utility functions
.env                # Environment variable file
.gitignore          # Git ignore file
config.py           # global variable configuration
Pipfile             # requirements file at pipenv format
Pipfile.loc         # requirements lock file
README.md           # 
requirements.txt    # requirements file at pip format
```

# How to use the code?
## Command Line
To perform the automated perspective correction _(let the algorithm decided which method is the best)_ of an image via command line use
```bash
python warp.py <image_path>
```

In case you want to apply a specific methodology, vanishing point or marker detection you may use respectively:
```bash
python warp_without_marker.py <image_path>
```
```bash
python warp_with_marker.py <image_path>
```

Each script will try to perform the correction _(if possible)_ and will save the image with a `_warp` suffix in the same folder.

## API
Once you have copied the Pericolo folder in your project you can import the `ImageWarper` via 
```python
from Pericolo import ImageWarper
```

You can then create an instance of `ImageWarper` to use it as follow
 ```python
iw = ImageWarper("path/to/image/to/warp")
auto_warped_img = iw.warp() # Automated warping
marker_warped_img = iw.warp_with_marker() # Aruco warping
vanishingp_warped_img = iw.warp_without_marker() # Vanishing Point Warping
result_log = iw.logs # Get the validation steps as a dictionnary
```

Further documentation of the `ImageWarper` class is proviede [HERE](docs/IMAGEWARPER.md)

# Configuration
To configure the `ImageWarper` behavior you can tweak the `config.py` file in the root folder

### RANSAC VARIABLES
> #### NUM_ITERATIONS (Vanishing Point only)
>Number of sampling iterations to perform during RANSAC

> #### TRAINING_THRESHOLD (Vanishing Point only)
> Maximum angle, between edgelet directions and the `edgelet_middle_point -> vanishing point` vectors, to consider edgelet as inliers and use their stengths to compute score **during RANSAC**

>#### COMPLIANCE_THRESHOLD (Vanishing Point only)
>Maximum angle, between edgelet directions and the `edgelet_middle_point -> vanishing point` vectors, to consider edgelet as inliers and use their stengths to compute score **during the inlier edgelet removal**

### CONFIDENCE INDEX
>#### MIN_RANSAC_EDGELETS (Vanishing Point only)
> Minimum edgelets number allowed to perform RANSAC


>#### FIRST_RANSAC_SCORE (Vanishing Point only)
> Minimum score allowed during the first RANSAC

>#### RANSAC_CONDITIONS_TO_VALID (Vanishing Point only)
> Ratio of condition to validated during RANSAC algorithm to consider it successful

>#### IOU_THRESHOLD (Aruco Marker only)
> Intersection over Union threshold between Aruco unenclosed detection and SIFT unenclosed detection to outperform

>#### IOU_THRESHOLD (Aruco Marker only)
> Inclusion Ratio threshold between Aruco unenclosed detection and SIFT enclosed detection to outperform

### ARUCO 
>#### MARKER_SIZE_IN_MM (Aruco Marker only)
> Size, in millimeters, of the printed marker

>#### ENCLOSED_MARKER (Aruco Marker only)
> Whether or not the marker used is enclosed

>#### MARKER_ID (Aruco Marker only)
> Aruco ID of the marker

### HOMOGRAPHY 
>#### CLIP
> Whether or not to clip the output warped image if it is too big

>#### CLIP_FACTOR
> By which factor to clip the image if `CLIP` is set as `True`

>#### ALLOW_WARPING_INTERUPT
> Allow the computation interruption if the output size is too big to be computed

>#### MAX_SIZE_ALLOWED
> Maximum size allowed before the process is interrupted

## SIFT DETECTION
>#### ENCLOSED_MARKER_FILE (Aruco Marker only)
> Name of the enclosed marker file. This file should be in the `markers` folder

>#### UNENCLOSED_MARKER_FILE (Aruco Marker only)
> Name of the unenclosed marker file. This file should be in the `markers` folder

>#### ENFORCE_SIFT_DETECTION (Aruco Marker only)
> Whether or not to use SIFT detection to confirm the Aruco detection

>#### MIN_MATCH_COUNT (Aruco Marker only)
> Minimum match number to consider SIFT detection success

>#### MODERATE_MATCH_COUNT (Aruco Marker only)
> Match number threshold to consider the SIFT detection has a moderate confidence

>#### MIN_MATCH_COUNT (Aruco Marker only)
> Match number threshold to consider the SIFT detection has a strong confidence

>#### CONFIDENCE_LEVEL (Aruco Marker only)
> Which confidence level to use to determine if SIFT detection is genuine
>
> 0. Use **Minimum** confidence Level
> 1. Use **Moderate** confidence Level
> 2. Use **Strong** confidence Level
> - Use negative value to not use any confidence level
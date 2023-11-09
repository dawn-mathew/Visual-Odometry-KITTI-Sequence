# Visual-Odometry-KITTI-Sequence

## Introduction
This repository offers a basic implementation of Visual Odometry using the KITTI Sequence dataset. It includes the dataset itself and provides visualization tools for comparing the results between predicted and ground truth trajectories.

## Repo Structure
```
├── KITTI_sequence_1
│   ├── calib.txt
│   ├── image_l
│   └── poses.txt
├── KITTI_sequence_2
│   ├── calib.txt
│   ├── image_l
│   └── poses.txt
├── lib
│   └── visualization
└── Visual_Odometery.py
```
<div style="text-align: justify;">

- KITTI_sequence_1 & KITTI_sequence_2 are independent datasets with their respective `calib.txt` and `poses.txt` files.
- `Calib.txt` contains the calibration matrix with the intrinsic and extrinsic parameters associated with each KITTI sequence. The calibration matrix generally encompasses parameters, including focal length, principal point coordinates, and lens distortion coefficients. Through the application of these parameters, computer vision algorithms can effectively delineate real-world objects to their respective positions within the 2D image plane, thereby facilitating tasks such as visual odometry.
- `Poses.txt` contains the ground truth data of the ego vehicle. This data is used to evaluate the accuracy of the predicted trajectory with the actual trajectory of the ego vehicle.
- `lib` comprises all requisite visualization tools for observing matched points between two successive frames in a sequence, as well as plotting tools designed to depict the predicted trajectory alongside the ground truth data.

</div>


## Dependencies
Install all of the following libraries for the main Visual Odometry file
```py
import os
import numpy as np
import cv2
from lib.visualization import plotting
from lib.visualization.video import play_trip
from tqdm import tqdm
```
Install all of the following libraries for plotting functions
```py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from bokeh.io import output_file, show
from bokeh.plotting import figure, ColumnDataSource
from bokeh.layouts import column, layout, gridplot
from bokeh.models import Div, WheelZoomTool
from bokeh.models.layouts import TabPanel, Tabs
```
## Results
### Visualizing Interest Point Matching
[Point_matching.webm](https://github.com/dawn-mathew/Visual-Odometry-KITTI-Sequence/assets/150279674/58cd8bdd-9b88-4fc4-98d6-735af228d50b)
- The video shows the various points of interest being matched between two consecutive frames from the KITTI_sequence_1 dataset. These matched points are then used to compute the predicted trajectory of the ego vehicle.
### Trajectory Visualization
![Dashboard](https://github.com/dawn-mathew/Visual-Odometry-KITTI-Sequence/assets/150279674/fcec0c92-4881-4a97-8566-874e6157d278)
- The image depicts the HTML dashboard created using the bokeh library. From the results plotted its understood that the predicted trajectory tends to deviate from the ground truth and result in a shorter/wider predicted vehicle trajectory. The error is also observed to incrases with the number of computed frames. Futher improvements have to be made with more complex approaches such as tracking immobile objects or landmarks such as tress or buildings and usnig its computed motion to correct the predicted trajectory of the ego vehicle.




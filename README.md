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

- KITTI_sequence_1 & KITTI_sequence_2 are idependent datasets with their respective calib.txt and poses.txt files
    - Calib.txt contain the caliberation matrix with the intrinsic and extrinsic parameters associated with each KITTI sequence. The calibration matrix generally encompasses parameters, including focal length, principal point coordinates, and lens distortion coefficients. Through the application of these parameters, computer vision algorithms can effectively delineate real-world objects to their respective positions within the 2D image plane, thereby facilitating tasks such as visual odometry.
- Poses.txt contains the ground truth data of the ego vehicle. This data is used evaluate the accuracy of the predicted trajectory with the actual trajectory of the ego vehicle.
- lib comprises all requisite visualization tools for observing matched points between two successive frames in a sequence, as well as plotting tools designed to depict the predicted trajectory alongside the ground truth data. 

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


[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [dill]: <https://github.com/joemccann/dillinger>
   [git-repo-url]: <https://github.com/joemccann/dillinger.git>
   [john gruber]: <http://daringfireball.net>
   [df1]: <http://daringfireball.net/projects/markdown/>
   [markdown-it]: <https://github.com/markdown-it/markdown-it>
   [Ace Editor]: <http://ace.ajax.org>
   [node.js]: <http://nodejs.org>
   [Twitter Bootstrap]: <http://twitter.github.com/bootstrap/>
   [jQuery]: <http://jquery.com>
   [@tjholowaychuk]: <http://twitter.com/tjholowaychuk>
   [express]: <http://expressjs.com>
   [AngularJS]: <http://angularjs.org>
   [Gulp]: <http://gulpjs.com>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>

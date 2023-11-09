import os
import numpy as np
import cv2

from lib.visualization import plotting
from lib.visualization.video import play_trip
from tqdm import tqdm

#............ // _load_calib  //...............
def _load_calib(filepath):
    with open(filepath, 'r') as f:
        params = np.fromstring(f.readline(), dtype=np.float64, sep=' ')
        P = np.reshape(params, (3, 4))
        K = P[0:3, 0:3]
    return K, P


#............ // _load_poses //...............
def _load_poses(filepath):
    poses = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            T = np.fromstring(line, dtype=np.float64, sep=' ')
            T = T.reshape(3, 4)
            T = np.vstack((T, [0, 0, 0, 1]))
            poses.append(T)
    return poses


#............ // _load_images //...............
def _load_images(filepath):
    image_paths = [os.path.join(filepath, file) for file in sorted(os.listdir(filepath))]
    return [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]


#............ // _form_transf //...............
def _form_transf(R,t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


#............ // get_matches //...............
def get_matches(i):
    kp1, des1 = orb.detectAndCompute(images[i - 1], None)
    kp2, des2 = orb.detectAndCompute(images[i], None)
    matches = flann.knnMatch(des1, des2, k=2)

    good = []
    try:
            for m, n in matches:
                if m.distance < 0.8 * n.distance:
                    good.append(m)
    except ValueError:
        pass

    
    draw_params = dict(matchColor = -1, # draw matches in green color
                 singlePointColor = None,
                 matchesMask = None, # draw only inliers
                 flags = 2)
    img3 = cv2.drawMatches(images[i], kp1, images[i-1],kp2, good ,None,**draw_params)
    cv2.imshow("image", img3)
    cv2.waitKey(200)

    q1 = np.float32([kp1[m.queryIdx].pt for m in good])
    q2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return q1, q2


#............ // get_poses //...............
def get_pose(q1,q2):
    E, _ = cv2.findEssentialMat(q1, q2, K, threshold=1)
    R, t = decomp_essential_mat(E, q1, q2)
    transformation_matrix = _form_transf(R, np.squeeze(t))
    return transformation_matrix

#............ // decomp_essential_mat //...............
def decomp_essential_mat(E, q1, q2):
    def sum_z_cal_relative_scale(R, t):
        T = _form_transf(R, t)
        P1 = np.matmul(np.concatenate((K, np.zeros((3, 1))), axis=1), T)
        hom_Q1 = cv2.triangulatePoints(P, P1, q1.T, q2.T)
        hom_Q2 = np.matmul(T, hom_Q1)

        uhom_Q1 = hom_Q1[:3, :] / hom_Q1[3, :]
        uhom_Q2 = hom_Q2[:3, :] / hom_Q2[3, :]

        sum_of_pos_z_Q1 = sum(uhom_Q1[2, :] > 0)
        sum_of_pos_z_Q2 = sum(uhom_Q2[2, :] > 0)

        relative_scale = np.mean(np.linalg.norm(uhom_Q1.T[:-1] - uhom_Q1.T[1:], axis=-1)/np.linalg.norm(uhom_Q2.T[:-1] - uhom_Q2.T[1:], axis=-1))
        return sum_of_pos_z_Q1 + sum_of_pos_z_Q2, relative_scale
        
    R1, R2, t = cv2.decomposeEssentialMat(E)
    t = np.squeeze(t)
    pairs = [[R1, t], [R1, -t], [R2, t], [R2, -t]]

    z_sums = []
    relative_scales = []
    for R, t in pairs:
        z_sum, scale = sum_z_cal_relative_scale(R, t)
        z_sums.append(z_sum)
        relative_scales.append(scale)
    
    right_pair_idx = np.argmax(z_sums)
    right_pair = pairs[right_pair_idx]
    relative_scale = relative_scales[right_pair_idx]
    R1, t = right_pair
    t = t * relative_scale
    return [R1, t]

# ....................// BODY // .......................
data_dir = "KITTI_sequence_1"

# VisualOdometery
K, P = _load_calib(os.path.join(data_dir, 'calib.txt'))
gt_poses = _load_poses(os.path.join(data_dir,"poses.txt"))
images = _load_images(os.path.join(data_dir,"image_l"))
orb = cv2.ORB_create(3000)
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(indexParams=index_params, searchParams=search_params)

play_trip(images)

gt_path = []
estimated_path = []

for i, gt_pose in enumerate(tqdm(gt_poses, unit="pose")):
    if i == 0:
         cur_pose = gt_pose
    else:
        q1, q2 = get_matches(i)
        transf = get_pose(q1, q2)
        cur_pose = np.matmul(cur_pose, np.linalg.inv(transf))
    
    gt_path.append((gt_pose[0, 3], gt_pose[2, 3]))
    estimated_path.append((cur_pose[0, 3], cur_pose[2, 3]))
    

plotting.visualize_paths(gt_path, estimated_path, "Visual Odometry", file_out=os.path.basename(data_dir) + ".html")

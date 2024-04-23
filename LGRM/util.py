import numpy as np
import tf.transformations as TF

WIDTH = 640
HEIGHT = 480

def PCA(PC):

    # Move to zero center & Find egienvalues, eigenvectors
    PC = PC - PC.mean(axis=0) # (H*W, 3)
    cov = np.matmul(PC.T, PC)
    eigvals, eigvec = np.linalg.eigh(cov)
    # Return the axis with the longest eignevalue
    return eigvals[-1], eigvec[-1]

def floor_removal(segment, max_iter=10000, thresh=1e-2):

    # Remove NaN elements
    H, W = segment.shape
    X = segment['x'].reshape(H*W, 1)
    Y = segment['y'].reshape(H*W, 1)
    Z = segment['z'].reshape(H*W, 1)
    PC = np.hstack([X, Y, Z])
    nan_indices = np.argwhere(np.isnan(PC.sum(1)))
    num_indices = np.argwhere(np.logical_not(np.isnan(PC.sum(1))))
    PC = np.delete(PC, nan_indices, axis=0)
    # Find cloud points from floor using a RANSAC
    best_eq = []
    best_inliers = []
    for idx in range(max_iter):

        plane_eq = np.random.random((4)) * 4 - 2.
        plane_eq[0] = (np.random.random() * 2 - 1.) * 1e-2
        plane_eq[1] = (np.random.random() * 2 - 1.) * 1e-2
        dist_pt = (plane_eq[0] * PC[:, 0] + plane_eq[1] * PC[:, 1] + plane_eq[2] * PC[:, 2] + plane_eq[3]) / np.sqrt(plane_eq[0]**2 + plane_eq[1]**2 + plane_eq[2]**2)
        inlier_indices = np.where(np.abs(dist_pt) <= thresh)[0]
        if len(inlier_indices) > len(best_inliers):
            best_eq = plane_eq
            best_inliers = inlier_indices
    # After RANSAC, remove the selected floor
    PC = np.delete(PC, best_inliers, axis=0)
    return PC

def vg_targets(pick_tl_x, pick_tl_y, pick_br_x, pick_br_y, place_x_ct, place_y_ct, cloud, clean):

    # Remove support floor of pick object
    cloud = cloud.reshape(HEIGHT, WIDTH)
    segment = cloud[pick_tl_y: pick_br_y, pick_tl_x: pick_br_x]
    segment = floor_removal(segment)
    # Compute picking coordinates by compute mean of segmented objects
    pick_x = segment[:, 0].mean() + 0.03 
    pick_y = segment[:, 1].mean() + 0.01 
    pick_z = max(segment[:, 2].mean(), 0.02) 
    pick_target = [float(pick_x), float(pick_y), float(pick_z)]
    # Compute placement coordinate
    clean = clean.reshape(HEIGHT, WIDTH)
    h_min = max(int(place_y_ct - 15), 0)
    h_max = min(int(place_y_ct + 15), 480)
    w_min = max(int(place_x_ct - 15), 0)
    w_max = min(int(place_x_ct + 15), 640)
    segment = cloud[h_min: h_max, w_min: w_max]
    place_x = np.nanmean(segment['x']) + 0.03 
    place_y = np.nanmean(segment['y']) + 0.04 
    place_z = max(float(pick_z), 0.02)
    place_target = [float(place_x), float(place_y), place_z]
    return pick_target, place_target

def vg_targets_grasp_only(pick_tl_x, pick_tl_y, pick_br_x, pick_br_y, cloud):

    # Remove support floor of pick object
    cloud = cloud.reshape(HEIGHT, WIDTH)
    segment = cloud[pick_tl_y: pick_br_y, pick_tl_x: pick_br_x]
    segment = floor_removal(segment)
    # Compute picking coordinates by compute mean of segmented objects
    pick_x = segment[:, 0].mean() + 0.015
    pick_y = segment[:, 1].mean()
    pick_z = max(segment[:, 2].mean(), -0.005)
    pick_target = [float(pick_x), float(pick_y), float(pick_z)]
    # Is this useful? I'm not sure ...
    return pick_target


def random_orientation_sample(current_ori, K=8, scale=5e-1):

    mat = TF.quaternion_matrix(np.array(current_ori))
    q_arr = []
    for _ in range(K):
        delta = scale * np.random.uniform(-1, 1, 3)
        delta_action = TF.euler_matrix(*delta)
        q_arr.append(TF.quaternion_from_matrix(np.matmul(delta_action, mat)))
    return q_arr








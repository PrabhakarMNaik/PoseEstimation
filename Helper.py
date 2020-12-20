import pickle
import numpy as np
import cv2
import os
from scipy.spatial.transform import Rotation as R

# Functions:
# save_obj
# load_obj
# ADD_score_norm(pt_cld, true_pose, pred_pose, diameter)
# ADD_score(pt_cld, true_pose, pred_pose, diameter)
# create_bounding_box(img, pose, pt_cld_data, intrinsic_matrix,color=(0,0,255))
# dataset_dir_structure(root_dir, class_names)
# get_rot_tra(rot_adr, tra_adr)
# fill_holes(idmask, umask, vmask)
# create_rendering(root_dir, intrinsic_matrix, obj, idx)
# create_rendering_eval(root_dir, intrinsic_matrix, obj, rigid_transformation)
###########################################################################################################
def save_obj(obj, name):
    """
    Create dictionaries for the training and testing data
    args: 1. Object to be stored
          2. Identifier for the object
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
###########################################################################################################

def load_obj(name):
    """
    Loads a pickle object
    args: name: name of the object
    returns: object stored in the file
    """
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)
###########################################################################################################

def ADD_score(pt_cld, true_pose, pred_pose, diameter):
    "Evaluation metric - ADD score"
    pred_pose[0:3, 0:3][np.isnan(pred_pose[0:3, 0:3])] = 1
    pred_pose[:, 3][np.isnan(pred_pose[:, 3])] = 0
    target = pt_cld @ true_pose[0:3, 0:3] + np.array(
        [true_pose[0, 3], true_pose[1, 3], true_pose[2, 3]])
    output = pt_cld @ pred_pose[0:3, 0:3] + np.array(
        [pred_pose[0, 3], pred_pose[1, 3], pred_pose[2, 3]])
    avg_distance = (np.linalg.norm(output - target))/pt_cld.shape[0]
    threshold = diameter * 0.1
    if avg_distance <= threshold:
        return 1
    else:
        return 0
###########################################################################################################

def create_bounding_box(img, pose, pt_cld_data, intrinsic_matrix,color=(0,0,255)):
    "Create a bounding box around the object"
    # 8 corner points of the ptcld data
    min_x, min_y, min_z = pt_cld_data.min(axis=0)
    max_x, max_y, max_z = pt_cld_data.max(axis=0)
    
    corners_3D = np.array([[max_x, min_y, min_z],
                        [max_x, min_y, max_z],
                        [min_x, min_y, max_z],
                        [min_x, min_y, min_z],
                        [max_x, max_y, min_z],
                        [max_x, max_y, max_z],
                        [min_x, max_y, max_z],
                        [min_x, max_y, min_z]])

    # convert these 8 3D corners to 2D points
    ones = np.ones((corners_3D.shape[0], 1))
    homogenous_coordinate = np.append(corners_3D, ones, axis=1)

    # Perspective Projection to obtain 2D coordinates for masks
    homogenous_2D = intrinsic_matrix @ (pose @ homogenous_coordinate.T)
    coord_2D = homogenous_2D[:2, :] / homogenous_2D[2, :]

    coord_2D = ((np.floor(coord_2D)).T).astype(int)
    # Draw lines between these 8 points
    if (coord_2D > 0).all():
        try:
            img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[1]), color, 3)
            img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[3]), color, 3)
            img = cv2.line(img, tuple(coord_2D[0]), tuple(coord_2D[4]), color, 3)
            img = cv2.line(img, tuple(coord_2D[1]), tuple(coord_2D[2]), color, 3)
            img = cv2.line(img, tuple(coord_2D[1]), tuple(coord_2D[5]), color, 3)
            img = cv2.line(img, tuple(coord_2D[2]), tuple(coord_2D[3]), color, 3)
            img = cv2.line(img, tuple(coord_2D[2]), tuple(coord_2D[6]), color, 3)
            img = cv2.line(img, tuple(coord_2D[3]), tuple(coord_2D[7]), color, 3)
            img = cv2.line(img, tuple(coord_2D[4]), tuple(coord_2D[7]), color, 3)
            img = cv2.line(img, tuple(coord_2D[4]), tuple(coord_2D[5]), color, 3)
            img = cv2.line(img, tuple(coord_2D[5]), tuple(coord_2D[6]), color, 3)
            img = cv2.line(img, tuple(coord_2D[6]), tuple(coord_2D[7]), color, 3)
        except OverflowError:
            pass
    return img

###########################################################################################################

def ADD_score_norm(pt_cld, true_pose, pred_pose, diameter):
    "Evaluation metric - ADD score"
    pred_pose[0:3, 0:3][np.isnan(pred_pose[0:3, 0:3])] = 1
    pred_pose[:, 3][np.isnan(pred_pose[:, 3])] = 0
    target = pt_cld @ true_pose[0:3, 0:3] + np.array(
        [true_pose[0, 3], true_pose[1, 3], true_pose[2, 3]])
    output = pt_cld @ pred_pose[0:3, 0:3] + np.array(
        [pred_pose[0, 3], pred_pose[1, 3], pred_pose[2, 3]])
    avg_distance = (np.linalg.norm(output - target))/pt_cld.shape[0]
    threshold = diameter * 0.1
    return avg_distance

###########################################################################################################

def dataset_dir_structure(root_dir, class_names):
    """
    Creates a directory structure for the classes of the objects
    present in the dataset.
    args: 1. root_dir: parent directory
          2. class_names: classes of samples in the dataset
    """
    try:
        for label in class_names:  # create directories to store data
            os.mkdir(root_dir + label + "/predicted_pose")
            os.mkdir(root_dir + label + "/ground_truth")
            os.mkdir(root_dir + label + "/ground_truth/IDmasks")
            os.mkdir(root_dir + label + "/ground_truth/Umasks")
            os.mkdir(root_dir + label + "/ground_truth/Vmasks")
            os.mkdir(root_dir + label + "/changed_background")
            os.mkdir(root_dir + label + "/pose_refinement")
            os.mkdir(root_dir + label + "/pose_refinement/real")
            os.mkdir(root_dir + label + "/pose_refinement/rendered")
    except FileExistsError:
        print("Directories already exist")

###########################################################################################################

def get_rot_tra(rot_adr, tra_adr):
    """
    Helper function to the read the rotation and translation file
        Args:
                rot_adr (str): path to the file containing rotation of an object
        tra_adr (str): path to the file containing translation of an object
        Returns:
                rigid transformation (np array): rotation and translation matrix combined
    """

    rot_matrix = np.loadtxt(rot_adr, skiprows=1)
    trans_matrix = np.loadtxt(tra_adr, skiprows=1)
    trans_matrix = np.reshape(trans_matrix, (3, 1))
    rigid_transformation = np.append(rot_matrix, trans_matrix, axis=1)

    return rigid_transformation

###########################################################################################################

def fill_holes(idmask, umask, vmask):
    """
    Helper function to fill the holes in id , u and vmasks
        Args:
                idmask (np.array): id mask whose holes you want to fill
        umask (np.array): u mask whose holes you want to fill
        vmask (np.array): v mask whose holes you want to fill
        Returns:
                filled_id_mask (np array): id mask with holes filled
        filled_u_mask (np array): u mask with holes filled
        filled_id_mask (np array): v mask with holes filled
    """
    idmask = np.array(idmask, dtype='float32')
    umask = np.array(umask, dtype='float32')
    vmask = np.array(vmask, dtype='float32')
    thr, im_th = cv2.threshold(idmask, 0, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    res = cv2.morphologyEx(im_th, cv2.MORPH_OPEN, kernel)
    im_th = cv2.bitwise_not(im_th)
    des = cv2.bitwise_not(res)
    mask = np.array(des-im_th, dtype='uint8')
    filled_id_mask = cv2.inpaint(idmask, mask, 5, cv2.INPAINT_TELEA)
    filled_u_mask = cv2.inpaint(umask, mask, 5, cv2.INPAINT_TELEA)
    filled_v_mask = cv2.inpaint(vmask, mask, 5, cv2.INPAINT_TELEA)

    return filled_id_mask, filled_u_mask, filled_v_mask

############################################################################################################

def create_rendering(root_dir, intrinsic_matrix, obj, idx):
    # helper function to help with creating renderings
    pred_pose_adr = root_dir + obj + \
        '/predicted_pose' + '/info_' + str(idx) + ".txt"
    rgb_values = np.loadtxt(root_dir + obj + '/object.xyz',
                            skiprows=1, usecols=(6, 7, 8))
    coords_3d = np.loadtxt(root_dir + obj + '/object.xyz',
                           skiprows=1, usecols=(0, 1, 2))
    ones = np.ones((coords_3d.shape[0], 1))
    homogenous_coordinate = np.append(coords_3d, ones, axis=1)
    rigid_transformation = np.loadtxt(pred_pose_adr)
    # Perspective Projection to obtain 2D coordinates
    homogenous_2D = intrinsic_matrix @ (
        rigid_transformation @ homogenous_coordinate.T)
    homogenous_2D[2, :][np.where(homogenous_2D[2, :] == 0)] = 1
    coord_2D = homogenous_2D[:2, :] / homogenous_2D[2, :]
    coord_2D = ((np.floor(coord_2D)).T).astype(int)
    rendered_image = np.zeros((480, 640, 3))
    x_2d = np.clip(coord_2D[:, 0], 0, 479)
    y_2d = np.clip(coord_2D[:, 1], 0, 639)
    rendered_image[x_2d, y_2d, :] = rgb_values
    temp = np.sum(rendered_image, axis=2)
    non_zero_indices = np.argwhere(temp > 0)
    min_x = non_zero_indices[:, 0].min()
    max_x = non_zero_indices[:, 0].max()
    min_y = non_zero_indices[:, 1].min()
    max_y = non_zero_indices[:, 1].max()
    cropped_rendered_image = rendered_image[min_x:max_x +
                                            1, min_y:max_y + 1, :]
    if cropped_rendered_image.shape[0] > 240 or cropped_rendered_image.shape[1] > 320:
        cropped_rendered_image = cv2.resize(np.float32(
            cropped_rendered_image), (320, 240), interpolation=cv2.INTER_AREA)
    return cropped_rendered_image


def create_rendering_eval(root_dir, intrinsic_matrix, obj, rigid_transformation):
    # helper function to help with creating renderings
    rgb_values = np.loadtxt(root_dir + obj + '/object.xyz',
                            skiprows=1, usecols=(6, 7, 8))
    coords_3d = np.loadtxt(root_dir + obj + '/object.xyz',
                           skiprows=1, usecols=(0, 1, 2))
    ones = np.ones((coords_3d.shape[0], 1))
    homogenous_coordinate = np.append(coords_3d, ones, axis=1)
    # Perspective Projection to obtain 2D coordinates
    homogenous_2D = intrinsic_matrix @ (
        rigid_transformation @ homogenous_coordinate.T)
    homogenous_2D[2, :][np.where(homogenous_2D[2, :] == 0)] = 1
    coord_2D = homogenous_2D[:2, :] / homogenous_2D[2, :]
    coord_2D = ((np.floor(coord_2D)).T).astype(int)
    rendered_img = np.zeros((480, 640, 3))
    x_2d = np.clip(coord_2D[:, 0], 0, 479)
    y_2d = np.clip(coord_2D[:, 1], 0, 639)
    rendered_img[x_2d, y_2d, :] = rgb_values
    temp = np.sum(rendered_img, axis=2)
    non_zero_indices = np.argwhere(temp > 0)
    min_x = non_zero_indices[:, 0].min()
    max_x = non_zero_indices[:, 0].max()
    min_y = non_zero_indices[:, 1].min()
    max_y = non_zero_indices[:, 1].max()
    cropped_rendered_img = rendered_img[min_x:max_x +
                                        1, min_y:max_y + 1, :]
    if cropped_rendered_img.shape[0] > 240 or cropped_rendered_img.shape[1] > 320:
        cropped_rendered_img = cv2.resize(np.float32(
            cropped_rendered_img), (320, 240), interpolation=cv2.INTER_AREA)
    return cropped_rendered_img

###########################################################################################################
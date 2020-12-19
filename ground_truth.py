from Helper import *
import os
import re
import cv2
import numpy as np
import random
import matplotlib.image as mpimg



def create_GT_masks(root_dir, background_dir, intrinsic_matrix,classes):
    """
    Helper function to create the Ground Truth ID,U and V masks
        Args:
        root_dir (str): path to the root directory of the dataset
        background_dir(str): path t
        intrinsic_matrix (array): matrix containing camera intrinsics
        classes (dict) : dictionary containing classes and their ids
        Saves the masks to their respective directories
    """
    list_all_images = load_obj(root_dir + "all_images_adr")
    training_images_idx = load_obj(root_dir + "train_images_indices")
    for i in range(len(training_images_idx)):
        img_adr = list_all_images[training_images_idx[i]]
        label = os.path.split(os.path.split(os.path.dirname(img_adr))[0])[1]
        regex = re.compile(r'\d+')
        idx = regex.findall(os.path.split(img_adr)[1])[0]

        if i % 1000 == 0:
            print(str(i) + "/" + str(len(training_images_idx)) + " finished!")

        image = cv2.imread(img_adr)
        ID_mask = np.zeros((image.shape[0], image.shape[1]))
        U_mask = np.zeros((image.shape[0], image.shape[1]))
        V_mask = np.zeros((image.shape[0], image.shape[1]))

        ID_mask_file = root_dir + label + \
            "/ground_truth/IDmasks/color" + str(idx) + ".png"
        U_mask_file = root_dir + label + \
            "/ground_truth/Umasks/color" + str(idx) + ".png"
        V_mask_file = root_dir + label + \
            "/ground_truth/Vmasks/color" + str(idx) + ".png"

        tra_adr = root_dir + label + "/data/tra" + str(idx) + ".tra"
        rot_adr = root_dir + label + "/data/rot" + str(idx) + ".rot"
        rigid_transformation = get_rot_tra(rot_adr, tra_adr)

        # Read point Point Cloud Data
        ptcld_file = root_dir + label + "/object.xyz"
        pt_cld_data = np.loadtxt(ptcld_file, skiprows=1, usecols=(0, 1, 2))
        ones = np.ones((pt_cld_data.shape[0], 1))
        homogenous_coordinate = np.append(pt_cld_data[:, :3], ones, axis=1)

        # Perspective Projection to obtain 2D coordinates for masks
        homogenous_2D = intrinsic_matrix @ (rigid_transformation @ homogenous_coordinate.T)
        coord_2D      = homogenous_2D[:2, :] / homogenous_2D[2, :]
        coord_2D      = ((np.floor(coord_2D)).T).astype(int)
        x_2d          = np.clip(coord_2D[:, 0], 0, 639)
        y_2d          = np.clip(coord_2D[:, 1], 0, 479)
        ID_mask[y_2d, x_2d] = classes[label]

        if i % 100 != 0:  # change background for every 99/100 images
            background_img_adr = background_dir + random.choice(os.listdir(background_dir))
            background_img     = cv2.imread(background_img_adr)
            background_img     = cv2.resize(background_img, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA)
            background_img[y_2d, x_2d, :] = image[y_2d, x_2d, :]
            background_adr     = root_dir + label + "/changed_background/color" + str(idx) + ".png"
            mpimg.imsave(background_adr, background_img)

        # Generate Ground Truth UV Maps
        centre = np.mean(pt_cld_data, axis=0)
        length = np.sqrt((centre[0]-pt_cld_data[:, 0])**2 + (centre[1] -
                                                             pt_cld_data[:, 1])**2 + (centre[2]-pt_cld_data[:, 2])**2)
        unit_vector = [(pt_cld_data[:, 0]-centre[0])/length, (pt_cld_data[:,
                                                                          1]-centre[1])/length, (pt_cld_data[:, 2]-centre[2])/length]
        U = 0.5 + (np.arctan2(unit_vector[2], unit_vector[0])/(2*np.pi))
        V = 0.5 - (np.arcsin(unit_vector[1])/np.pi)
        U_mask[y_2d, x_2d] = U
        V_mask[y_2d, x_2d] = V

        # Saving ID, U and V masks after using the fill holes function
        ID_mask, U_mask, V_mask = fill_holes(ID_mask, U_mask, V_mask)
        cv2.imwrite(ID_mask_file, ID_mask)
        mpimg.imsave(U_mask_file, U_mask, cmap='gray')
        mpimg.imsave(V_mask_file, V_mask, cmap='gray')
        
#         try:
#             os.mkdir("UV_XYZ")
#         except FileExistsError:
#             pass
       
#         if i%100 == 0:
#             print("U:", np.shape(U_mask))
#             print("V:", np.shape(V_mask))
            
#             cv2.imwrite(os.path.join("UV_XYZ" , "U_" + str(i) +".jpg"), U_mask)
#             cv2.imwrite(os.path.join("UV_XYZ" , "V_" + str(i) +".jpg"), V_mask)

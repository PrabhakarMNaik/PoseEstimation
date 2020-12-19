import os
import numpy as np
import re
import cv2
import torch
from torch.utils.data import Dataset
from Helper import *

################################ Change Superclass ##################################
class PoseRefinerDataset(Dataset):

    """
    Args:
        root_dir (str): path to the dataset directory
        classes (dict): dictionary containing classes as key  
        transform : Transforms for input image
            """

    def __init__(self, root_dir, classes=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = classes
        self.list_all_images = load_obj(root_dir + "all_images_adr")
        self.training_images_idx = load_obj(root_dir + "train_images_indices")

    def __len__(self):
        return len(self.training_images_idx)

    def __getitem__(self, i):
        img_adr = self.list_all_images[self.training_images_idx[i]]
        label = os.path.split(os.path.split(os.path.dirname(img_adr))[0])[1]
        regex = re.compile(r'\d+')
        idx = regex.findall(os.path.split(img_adr)[1])[0]
        image = cv2.imread(self.root_dir + label +
                           '/pose_refinement/real/color' + str(idx) + ".png")
        rendered = cv2.imread(
            self.root_dir + label + '/pose_refinement/rendered/color' + str(idx) + ".png", cv2.IMREAD_GRAYSCALE)
        rendered = cv2.cvtColor(rendered.astype('uint8'), cv2.COLOR_GRAY2RGB)
        true_pose = get_rot_tra(self.root_dir + label + '/data/rot' + str(idx) + ".rot",
                                self.root_dir + label + '/data/tra' + str(idx) + ".tra")
        pred_pose_adr = self.root_dir + label + \
            '/predicted_pose/info_' + str(idx) + ".txt"
        pred_pose = np.loadtxt(pred_pose_adr)
        if self.transform:
            image = self.transform(image)
            rendered = self.transform(rendered)
        return label, image, rendered, true_pose, pred_pose
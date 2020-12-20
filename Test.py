import copy
import time
import argparse
import torch.nn as nn
from torchvision import transforms, utils
from Correspondance_Network import UNet
import re
import os
from Helper import *
import cv2
import torch
from PoseRefinement import *
from scipy.spatial.transform import Rotation as R
import matplotlib.image as mpimg

def test():
    parser = argparse.ArgumentParser(
        description='Script to create the Ground Truth masks')
    parser.add_argument("--root_dir", default="LineMOD_Dataset/",
                        help="path to dataset directory")
    args, unknown = parser.parse_known_args()

    root_dir = args.root_dir
    fx = 572.41140
    px = 325.26110
    fy = 573.57043
    py = 242.04899
    intrinsic_matrix = np.array([[fx, 0, px], [0, fy, py], [0, 0, 1]])

    classes = {'ape': 1, 'benchviseblue': 2, 'cam': 3, 'can': 4, 'cat': 5, 'driller': 6,
               'duck': 7, 'eggbox': 8, 'glue': 9, 'holepuncher': 10, 'iron': 11, 'lamp': 12, 'phone': 13}

    score_card = {'ape': 0, 'benchviseblue': 0, 'cam': 0, 'can': 0, 'cat': 0, 'driller': 0,
                  'duck': 0, 'eggbox': 0, 'glue': 0, 'holepuncher': 0, 'iron': 0, 'lamp': 0, 'phone': 0}

    instances = {'ape': 0, 'benchviseblue': 0, 'cam': 0, 'can': 0, 'cat': 0, 'driller': 0,
                 'duck': 0, 'eggbox': 0, 'glue': 0, 'holepuncher': 0, 'iron': 0, 'lamp': 0, 'phone': 0}

    transform = transforms.Compose([transforms.ToPILImage(mode=None),
                                    transforms.Resize(size=(224, 224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    correspondence_block = UNet(n_channels=3, out_channels_id=14,
                                     out_channels_uv=256, bilinear=True)
    # load the best weights from the training loop
    correspondence_block.load_state_dict(torch.load(
        'correspondence_block.pt', map_location=torch.device('cpu')))
    pose_refiner = Pose_Refiner()
    # load the best weights from the training loop
    pose_refiner.load_state_dict(torch.load(
        'pose_refiner.pt', map_location=torch.device('cpu')))

    correspondence_block.cuda()
    pose_refiner.cuda()
    pose_refiner.eval()
    correspondence_block.eval()

    list_all_images = load_obj(root_dir + "all_images_adr")
    testing_images_idx = load_obj(root_dir + "test_images_indices")

    regex = re.compile(r'\d+')
    upsampled = nn.Upsample(size=[240, 320], mode='bilinear', align_corners=False)
    total_score = 0

    try:
        os.mkdir("Evaluated_Images")
    except FileExistsError:
        pass
    start = time.time()

    num_samples = 500

    for i in range(500):

        img_adr = list_all_images[testing_images_idx[i]]
        label = os.path.split(os.path.split(os.path.dirname(img_adr))[0])[1]
        idx = regex.findall(os.path.split(img_adr)[1])[0]

        tra_adr = root_dir + label + "/data/tra" + str(idx) + ".tra"
        rot_adr = root_dir + label + "/data/rot" + str(idx) + ".rot"
        true_pose = get_rot_tra(rot_adr, tra_adr)

        test_img = cv2.imread(img_adr)
        img_for_bounding_box = copy.deepcopy(test_img)
        test_img = cv2.resize(
            test_img, (test_img.shape[1]//2, test_img.shape[0]//2), interpolation=cv2.INTER_AREA)

        test_img = torch.from_numpy(test_img).type(torch.double)
        test_img = test_img.transpose(1, 2).transpose(0, 1)

        if len(test_img.shape) != 4:
            test_img = test_img.view(
                1, test_img.shape[0], test_img.shape[1], test_img.shape[2])

        # pass through correspondence block
        idmask_pred, umask_pred, vmask_pred = correspondence_block(
            test_img.float().cuda())

        # convert the masks to 240,320 shape
        temp = torch.argmax(idmask_pred, dim=1).squeeze().cpu()
        upred = torch.argmax(umask_pred, dim=1).squeeze().cpu()
        vpred = torch.argmax(vmask_pred, dim=1).squeeze().cpu()
        coord_2d = (temp == classes[label]).nonzero(as_tuple=True)
        if coord_2d[0].nelement() != 0:  # label is detected in the image
            coord_2d = torch.cat((coord_2d[0].view(
                coord_2d[0].shape[0], 1), coord_2d[1].view(coord_2d[1].shape[0], 1)), 1)
            uvalues = upred[coord_2d[:, 0], coord_2d[:, 1]]
            vvalues = vpred[coord_2d[:, 0], coord_2d[:, 1]]
            dct_keys = torch.cat((uvalues.view(-1, 1), vvalues.view(-1, 1)), 1)
            dct_keys = tuple(dct_keys.numpy())
            dct = load_obj(root_dir + label + "/UV-XYZ_mapping")
            mapping_2d = []
            mapping_3d = []
            for count, (u, v) in enumerate(dct_keys):
                if (u, v) in dct:
                    mapping_2d.append(np.array(coord_2d[count]))
                    mapping_3d.append(dct[(u, v)])

            # PnP needs atleast 6 unique 2D-3D correspondences to run
            if len(mapping_2d) >= 6 or len(mapping_3d) >= 6:
                _, rvecs, tvecs, inliers = cv2.solvePnPRansac(np.array(mapping_3d, dtype=np.float32),
                                                              np.array(mapping_2d, dtype=np.float32), intrinsic_matrix, distCoeffs=None,
                                                              iterationsCount=150, reprojectionError=1.0, flags=cv2.SOLVEPNP_P3P)
                rot, _ = cv2.Rodrigues(rvecs, jacobian=None)
                pred_pose = np.append(rot, tvecs, axis=1)

            else:  # save an empty file
                pred_pose = np.zeros((3, 4))

            min_x = coord_2d[:, 0].min()
            max_x = coord_2d[:, 0].max()
            min_y = coord_2d[:, 1].min()
            max_y = coord_2d[:, 1].max()
            img = test_img.squeeze().transpose(1, 2).transpose(0, 2)
            obj_img = img[min_x:max_x+1, min_y:max_y+1, :]
            # saving in the correct format using upsampling
            obj_img = obj_img.transpose(0, 1).transpose(0, 2).unsqueeze(dim=0)
            obj_img = upsampled(obj_img)
            obj_img = obj_img.squeeze().transpose(0, 2).transpose(0, 1)
            obj_img = transform(torch.as_tensor(obj_img, dtype=torch.float32))
            # create rendering for an object
            cropped_rendered_img = create_rendering_eval(
                root_dir, intrinsic_matrix, label, pred_pose)
            rendered_img = torch.from_numpy(cropped_rendered_img)
            rendered_img = rendered_img.unsqueeze(dim=0)
            rendered_img = rendered_img.transpose(1, 3).transpose(2, 3)
            rendered_img = upsampled(rendered_img)
            rendered_img = rendered_img.squeeze()
            rendered_img = transform(torch.as_tensor(
                rendered_img, dtype=torch.float32))

            if len(rendered_img.shape) != 4:
                rendered_img = rendered_img.view(
                    1, rendered_img.shape[0], rendered_img.shape[1], rendered_img.shape[2])

            if len(obj_img.shape) != 4:
                obj_img = obj_img.view(
                    1, obj_img.shape[0], obj_img.shape[1],  obj_img.shape[2])
            pred_pose = (torch.from_numpy(pred_pose)).unsqueeze(0)

            # pose refinement to get final output
            xy, z, rot = pose_refiner(obj_img.float().cuda(),
                                      rendered_img.float().cuda(), pred_pose)
            # below 2 lines are for outliers only - edge case                          
            rot[torch.isnan(rot)] = 1  # take care of NaN and inf values
            rot[rot == float("Inf")] = 1
            xy[torch.isnan(xy)] = 0
            z[torch.isnan(z)] = 0
            # convert R quarternion to rotational matrix
            rot = (R.from_quat(rot.detach().cpu().numpy())).as_matrix()
            pred_pose = pred_pose.squeeze().numpy()
            # update predicted pose
            xy = xy.squeeze()
            pred_pose[0:3, 0:3] = rot
            pred_pose[0, 3] = xy[0]
            pred_pose[1, 3] = xy[1]
            pred_pose[2, 3] = z

            diameter = np.loadtxt(root_dir + label + "/distance.txt")
            ptcld_file = root_dir + label + "/object.xyz"
            pt_cld = np.loadtxt(ptcld_file, skiprows=1, usecols=(0, 1, 2))
            test_rendered = create_bounding_box(img_for_bounding_box, pred_pose, pt_cld, intrinsic_matrix)
            test_rendered = create_bounding_box(test_rendered, true_pose, pt_cld, intrinsic_matrix, color = (0,255,0))

            score = ADD_score(pt_cld, true_pose, pred_pose, diameter)
            float_score = ADD_score_norm(pt_cld, true_pose, pred_pose, diameter)
            path = "Evaluated_Images/" + str(float_score) + "_score_" + str(i) +'.png'
            cv2.imwrite(path, test_rendered)
            total_score += score
            score_card[label] += score

        else:
            score_card[label] += 0

        instances[label] += 1

    print("Average Evaluation Time:", (time.time() - start)/num_samples)
    print("ADD Score for all testing images is: ",total_score/num_samples)
    float_score = [1 - score for score in float_score]
import os
import numpy as np
import re
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from Helper import *
from LineMOD import *
from scipy.spatial.transform import Rotation as R
from Correspondance_Network import UNet
from torchvision import transforms, utils
import matplotlib.image as mpimg
from torchvision import models
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

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

########################################################################################################

class Pose_Refiner(nn.Module):

    def __init__(self):
        super(Pose_Refiner, self).__init__()
        self.feature_extractor_image = nn.Sequential(*list(models.resnet18(pretrained=True,
                                                                           progress=True).children())[:9])
        self.feature_extractor_rendered = nn.Sequential(*list(models.resnet18(pretrained=True,
                                                                              progress=True).children())[:9])
        self.fc_xyhead_1 = nn.Linear(512, 253)
        self.fc_xyhead_2 = nn.Linear(256, 2)
        self.fc_zhead = nn.Sequential(nn.Linear(512, 256),
                                      nn.ReLU(),
                                      nn.Linear(256, 1))
        self.fc_Rhead_1 = nn.Linear(512, 252)
        self.fc_Rhead_2 = nn.Linear(256, 4)

        self.relu_layer = nn.ReLU()

    def _initialize_weights(self):
        # weight initialization
        nn.init.constant_(self.fc_xyhead_1.weight, 0.)
        nn.init.constant_(self.fc_xyhead_1.bias, 0.)

        weights = torch.zeros((2, 256))
        weights[0, 253] = torch.tensor(1.)
        weights[1, 254] = torch.tensor(1.)
        self.fc_xyhead_2.weight = nn.Parameter(weights)
        nn.init.constant_(self.fc_xyhead_2.bias, 0.)

        nn.init.constant_(self.fc_zhead.weight, 0.)
        nn.init.constant_(self.fc_zhead.bias, 0.)

        nn.init.constant_(self.fc_Rhead_1.weight, 0.)
        nn.init.constant_(self.fc_Rhead_1.bias, 0.)

        rand_weights = torch.zeros((4, 256))
        rand_weights[0, 252] = torch.tensor(1.)
        rand_weights[1, 253] = torch.tensor(1.)
        rand_weights[2, 254] = torch.tensor(1.)
        rand_weights[3, 255] = torch.tensor(1.)
        self.fc_Rhead_2.weight = nn.Parameter(weights)
        nn.init.constant_(self.fc_Rhead_2.bias, 0.)

    def forward(self, image, rendered, pred_pose, bs=1):
        # extracting the feature vector f
        f_image = self.feature_extractor_image(image)
        f_rendered = self.feature_extractor_rendered(rendered)
        f_image = f_image.view(bs, -1)
        f_image = self.relu_layer(f_image)
        f_rendered = f_rendered.view(bs, -1)
        f_rendered = self.relu_layer(f_rendered)
        f = f_image - f_rendered
        print(f.size())
        # Z refinement head
        z = self.fc_zhead(f)

        # XY refinement head
        f_xy1 = self.fc_xyhead_1(f)
        f_xy1 = self.relu_layer(f_xy1)
        x_pred = np.reshape(pred_pose[:, 0, 3], (bs, -1))
        y_pred = np.reshape(pred_pose[:, 1, 3], (bs, -1))
        f_xy1 = torch.cat((f_xy1, x_pred.float().cuda()), 1)
        f_xy1 = torch.cat((f_xy1, y_pred.float().cuda()), 1)
        f_xy1 = torch.cat((f_xy1, z), 1)
        xy = self.fc_xyhead_2(f_xy1.cuda())

        # Rotation head
        f_r1 = self.fc_Rhead_1(f)
        f_r1 = self.relu_layer(f_r1)
        r = R.from_matrix(pred_pose[:, 0:3, 0:3])
        r = r.as_quat()
        r = np.reshape(r, (bs, -1))
        f_r1 = torch.cat(
            (f_r1, torch.from_numpy(r).float().cuda()), 1)
        rot = self.fc_Rhead_2(f_r1)

        return xy, z, rot
    
############################################################################################################

def create_refinement_inputs(root_dir, classes, intrinsic_matrix):
    correspondence_block = UNet(
        n_channels=3, out_channels_id=14, out_channels_uv=256, bilinear=True)
    correspondence_block.cuda()
    correspondence_block.load_state_dict(torch.load(
        'correspondence_block.pt', map_location=torch.device('cpu')))

    train_data = LineMODDataset(root_dir, 
                                classes=classes,
                                transform=transforms.Compose([transforms.ToTensor()]))

    upsampled = nn.Upsample(size=[240, 320], mode='bilinear',align_corners=False)

    regex = re.compile(r'\d+')
    count = 0
    
    for i in range(len(train_data)):
        if i % 1000 == 0:
            print(str(i) + "/" + str(len(train_data)) + " finished!")
        img_adr, img, _, _, _ = train_data[i]

        label = os.path.split(os.path.split(os.path.dirname(img_adr))[0])[1]
        idx = regex.findall(os.path.split(img_adr)[1])[0]
        adr_rendered = root_dir + label + \
            "/pose_refinement/rendered/color" + str(idx) + ".png"
        adr_img = root_dir + label + \
            "/pose_refinement/real/color" + str(idx) + ".png"
        # find the object in the image using the idmask
        img = img.view(1, img.shape[0], img.shape[1], img.shape[2])
        idmask_pred, _, _ = correspondence_block(img.cuda())
        idmask = torch.argmax(idmask_pred, dim=1).squeeze().cpu()
        coord_2d = (idmask == classes[label]).nonzero(as_tuple=True)
        if coord_2d[0].nelement() != 0:
            coord_2d = torch.cat((coord_2d[0].view(
                coord_2d[0].shape[0], 1), coord_2d[1].view(coord_2d[1].shape[0], 1)), 1)
            min_x = coord_2d[:, 0].min()
            max_x = coord_2d[:, 0].max()
            min_y = coord_2d[:, 1].min()
            max_y = coord_2d[:, 1].max()
            img = img.squeeze().transpose(1, 2).transpose(0, 2)
            obj_img = img[min_x:max_x+1, min_y:max_y+1, :]
            # saving in the correct format using upsampling
            obj_img = obj_img.transpose(0, 1).transpose(0, 2).unsqueeze(dim=0)
            obj_img = upsampled(obj_img)
            obj_img = obj_img.squeeze().transpose(0, 2).transpose(0, 1)
            mpimg.imsave(adr_img, obj_img.squeeze().numpy())

            # create rendering for an object
            cropped_rendered_image = create_rendering(
                root_dir, intrinsic_matrix, label, idx)
            rendered_img = torch.from_numpy(cropped_rendered_image)
            rendered_img = rendered_img.unsqueeze(dim=0)
            rendered_img = rendered_img.transpose(1, 3).transpose(2, 3)
            rendered_img = upsampled(rendered_img)
            rendered_img = rendered_img.squeeze().transpose(0, 2).transpose(0, 1)
            mpimg.imsave(adr_rendered, rendered_img.numpy())

        else:  # object not present in idmask prediction
            count += 1
            mpimg.imsave(adr_rendered, np.zeros((240, 320)))
            mpimg.imsave(adr_img, np.zeros((240, 320)))
    print("Number of outliers: ", count)
    
########################################################################################################

def train_pose_refinement(root_dir, classes, epochs=5):
    
    train_data = PoseRefinerDataset(root_dir, classes=classes,
                                    transform=transforms.Compose([
                                        transforms.ToPILImage(mode=None),
                                        transforms.Resize(size=(224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [
                                                             0.229, 0.224, 0.225]),
                                        transforms.ColorJitter(
                                            brightness=0, contrast=0, saturation=0, hue=0)
                                    ]))

    pose_refiner = Pose_Refiner()
    pose_refiner.cuda()
    # freeze resnet
    # pose_refiner.feature_extractor[0].weight.requires_grad = False

    batch_size = 2
    num_workers = 0
    valid_size = 0.2
    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=train_sampler, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                               sampler=valid_sampler, num_workers=num_workers)

    optimizer = optim.Adam(pose_refiner.parameters(),
                           lr=1.5e-4, weight_decay=3e-5)

    # number of epochs to train the model
    n_epochs = epochs

    valid_loss_min = np.Inf  # track change in validation loss
    for epoch in range(1, n_epochs+1):
        torch.cuda.empty_cache()
        print("----- Epoch Number: ", epoch, "--------")

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        ###################
        # train the model #
        ###################
        pose_refiner.train()
        for label, image, rendered, true_pose, pred_pose in train_loader:
            # move tensors to GPU
            image, rendered = image.cuda(), rendered.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            xy, z, rot = pose_refiner(image, rendered, pred_pose, batch_size)
            # convert rot quarternion to rotational matrix
            rot[torch.isnan(rot)] = 1  # take care of NaN and inf values
            rot[rot == float("Inf")] = 1
            xy[torch.isnan(xy)] == 0
            z[torch.isnan(z)] == 0

            rot = torch.tensor(
                (R.from_quat(rot.detach().cpu().numpy())).as_matrix())
            # update predicted pose
            pred_pose[:, 0:3, 0:3] = rot
            pred_pose[:, 0, 3] = xy[:, 0]
            pred_pose[:, 1, 3] = xy[:, 1]
            pred_pose[:, 2, 3] = z.squeeze()
            # fetch point cloud data
            pt_cld = fetch_ptcld_data(root_dir, label, batch_size)
            # calculate the batch loss
            loss = Matching_loss(pt_cld, true_pose, pred_pose, batch_size)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()

        ######################
        # validate the model #
        ######################
        pose_refiner.eval()
        for label, image, rendered, true_pose, pred_pose in valid_loader:
            # move tensors to GPU
            image, rendered = image.cuda(), rendered.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            xy, z, rot = pose_refiner(image, rendered, pred_pose, batch_size)
            rot[torch.isnan(rot)] = 1  # take care of NaN and inf values
            rot[rot == float("Inf")] = 1            
            xy[torch.isnan(xy)] == 0
            z[torch.isnan(z)] == 0
            # convert R quarternion to rotational matrix
            rot = torch.tensor(
                (R.from_quat(rot.detach().cpu().numpy())).as_matrix())
            # update predicted pose
            pred_pose[:, 0:3, 0:3] = rot
            pred_pose[:, 0, 3] = xy[:, 0]
            pred_pose[:, 1, 3] = xy[:, 1]
            pred_pose[:, 2, 3] = z.squeeze()
            # fetch point cloud data
            pt_cld = fetch_ptcld_data(root_dir, label, batch_size)
            # calculate the batch loss
            loss = Matching_loss(pt_cld, true_pose, pred_pose, batch_size)
            # update average validation loss
            valid_loss += loss.item()

        # calculate average losses
        train_loss = train_loss/len(train_loader.sampler)
        valid_loss = valid_loss/len(valid_loader.sampler)

        # print training/validation statistics
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min, valid_loss))
            torch.save(pose_refiner.state_dict(), 'pose_refiner.pt')
            valid_loss_min = valid_loss

########################################################################################################
def fetch_ptcld_data(root_dir, label, bs):
    # detch pt cld data for batchsize
    pt_cld_data = []
    for i in range(bs):
        obj_dir = root_dir + label[i] + "/object.xyz"
        pt_cld = np.loadtxt(obj_dir, skiprows=1, usecols=(0, 1, 2))
        index = np.random.choice(pt_cld.shape[0], 3000, replace=False)
        pt_cld_data.append(pt_cld[index, :])
    pt_cld_data = np.stack(pt_cld_data, axis=0)
    return pt_cld_data

########################################################################################################
# no. of points is always 3000
def Matching_loss(pt_cld_rand, true_pose, pred_pose, bs, training=True):

    total_loss = torch.tensor([0.])
    total_loss.requires_grad = True
    for i in range(0, bs):
        pt_cld = pt_cld_rand[i, :, :].squeeze()
        TP = true_pose[i, :, :].squeeze()
        PP = pred_pose[i, :, :].squeeze()
        target = torch.tensor(pt_cld) @ TP[0:3, 0:3] + torch.cat(
            (TP[0, 3].view(-1, 1), TP[1, 3].view(-1, 1), TP[2, 3].view(-1, 1)), 1)
        output = torch.tensor(pt_cld) @ PP[0:3, 0:3] + torch.cat(
            (PP[0, 3].view(-1, 1), PP[1, 3].view(-1, 1), PP[2, 3].view(-1, 1)), 1)
        loss = (torch.abs(output - target).sum())/3000
        if loss < 100:
            total_loss = total_loss + loss
        else:  # so that loss isn't NaN
            total_loss = total_loss + torch.tensor([100.])

    return total_loss  
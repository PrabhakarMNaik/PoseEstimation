import os
import numpy as np
from Helper import save_obj


def create_UV_XYZ_dictionary(root_dir):

    classes = ['ape', 'benchviseblue', 'can', 'cat', 'driller', 'duck', 'glue', 'holepuncher',
               'iron', 'lamp', 'phone', 'cam', 'eggbox']
    # create a dictionary for UV to XYZ correspondence
    for label in classes:
        ptcld_file = root_dir + label + "/object.xyz"
        pt_cld_data = np.loadtxt(ptcld_file, skiprows=1, usecols=(0, 1, 2))
        # calculate u and v coordinates from the xyz point cloud file
        centre = np.mean(pt_cld_data, axis=0)
        length = np.sqrt((centre[0]-pt_cld_data[:, 0])**2 + (centre[1] -
                                                             pt_cld_data[:, 1])**2 + (centre[2]-pt_cld_data[:, 2])**2)
        unit_vector = [(pt_cld_data[:, 0]-centre[0])/length, (pt_cld_data[:,
                                                                          1]-centre[1])/length, (pt_cld_data[:, 2]-centre[2])/length]
        u_coord = 0.5 + (np.arctan2(unit_vector[2], unit_vector[0])/(2*np.pi))
        v_coord = 0.5 - (np.arcsin(unit_vector[1])/np.pi)
        u_coord = (u_coord * 255).astype(int)
        v_coord = (v_coord * 255).astype(int)
        # save the mapping as a pickle file
        dct = {}
        for u, v, xyz in zip(u_coord, v_coord, pt_cld_data):
            key = (u, v)
            if key not in dct:
                dct[key] = xyz
        save_obj(dct, root_dir + label + "/UV-XYZ_mapping")
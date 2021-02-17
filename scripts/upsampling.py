#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2021 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
# External libraries
import os
import pandas as pd
import numpy as np
from shutil import copyfile
import uuid
from PIL import Image
from multiprocessing.pool import ThreadPool
# AUCMEDI libraries
from aucmedi import Image_Augmentation
from aucmedi.data_processing.io_data import image_loader

#-----------------------------------------------------#
#                   Configurations                    #
#-----------------------------------------------------#
# Provide pathes to imaging and annotation data
# path_riadd = "/home/mudomini/data/RIADD/Training_Set/"
path_riadd = "/storage/riadd2021/Training_Set/"

# Provide path to upsampled directory
# path_aug = "/home/mudomini/data/RIADD/Upsampled_Set/"
path_aug = "/storage/riadd2021/Upsampled_Set/"

# Minimum number of occurences for a label combination
N_class = 100
N_pair = 1

# Subdirectories
path_images = os.path.join(path_riadd, "Training")
path_csv = os.path.join(path_riadd, "RFMiD_Training_Labels.csv")

#-----------------------------------------------------#
#              Identify Class Frequency               #
#-----------------------------------------------------#
dt = pd.read_csv(path_csv, sep=",")

#-----------------------------------------------------#
#           Identify rare Label Combinations          #
#-----------------------------------------------------#
def analyse_classes(dt):
    class_freq = dt.iloc[:, 1:].sum(axis=0).to_dict()

    print("-------------------------------------------------------------------")
    n_diseased = (dt["Disease_Risk"] == 1).sum()
    for c in class_freq:
        print(c, class_freq[c], (class_freq[c] / n_diseased))
    print(dt)

    labels_pairings = {}
    labels_pairings_ohe = {}
    col_list = dt.columns[1:]
    for index, row in dt.iterrows():
        pair = []
        for j, col in enumerate(col_list):
            if row[j+1] == 1 : pair.append(col)
        if not any(class_freq[c] < N_class for c in pair) : continue
        key = "|".join(pair)
        if key in labels_pairings : labels_pairings[key].append(str(row["ID"]))
        else:
            labels_pairings[key] = [str(row["ID"])]
            labels_pairings_ohe[key] = row[1:].tolist()

    return dt, class_freq, labels_pairings, labels_pairings_ohe

#-----------------------------------------------------#
#              Create Image Augmentation              #
#-----------------------------------------------------#
# Initialize image augmentation object
img_aug = Image_Augmentation(flip=True, rotate=True, brightness=True,
                             contrast=True, saturation=True, hue=True, scale=False,
                             crop=False, grid_distortion=False, compression=False,
                             gaussian_noise=False, gaussian_blur=False,
                             downscaling=False, gamma=False, elastic_transform=False)
# Image Augmentation function
def perform_augmentation(index_list, pair):
    # Randomly select an index
    ir = np.random.choice(len(index_list), 1, replace=True)
    index = index_list[ir[0]]
    # Obtain image
    img = image_loader(index, path_images_aug, image_format="png",
                       grayscale=False)
    # Perform augmentation
    img_new = img_aug.apply(img)
    # Generate new index
    index_new = "aug_" + index + "." + str(uuid.uuid4())
    # Store augmented image to upsampling directory
    path_img = os.path.join(path_images_aug, index_new + ".png")
    pil_im = Image.fromarray(img_new)
    pil_im.save(path_img)
    # Return new image name
    return index_new

#-----------------------------------------------------#
#                 Perform Up-Sampling                 #
#-----------------------------------------------------#
# Create upsampled directory
if not os.path.exists(path_aug) : os.mkdir(path_aug)
path_images_aug = os.path.join(path_aug, "images")
if not os.path.exists(path_images_aug) : os.mkdir(path_images_aug)

# Copy all available images to Upsampled Set at first
for file in os.listdir(path_images):
    copyfile(os.path.join(path_images, file),
             os.path.join(path_images_aug, file))

# Iterative up-sampling approach
class_freq = dt.iloc[:, 1:].sum(axis=0).to_dict()
while any(class_freq[c] < N_class for c in class_freq):
    dt, class_freq, labels_pairings, labels_pairings_ohe = analyse_classes(dt)

    for j in range(0, N_pair):
        # Obtain label pairing list
        wk_list = [(labels_pairings[pair], pair) for pair in labels_pairings]
        # Create augmentated image
        with ThreadPool(92) as pool:
            list_newIndicies = pool.starmap(perform_augmentation, wk_list)
        # Update data
        for i, new_index in enumerate(list_newIndicies):
            pair = wk_list[i][1]
            new_entry = [new_index] + labels_pairings_ohe[pair]
            dt.loc[len(dt)] = new_entry
            labels_pairings[pair].append(new_index)

print("-------------------------------------------------------------------")
n_diseased = (dt["Disease_Risk"] == 1).sum()
for c in class_freq:
    print(c, class_freq[c], (class_freq[c] / n_diseased))
print(dt)

# Store current data CSV to disk
dt.to_csv(os.path.join(path_aug, "data.csv"), sep=",", header=True, index=False)

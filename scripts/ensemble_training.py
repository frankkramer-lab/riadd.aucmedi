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
import json
import pandas as pd
# AUCMEDI libraries
from aucmedi import input_interface, DataGenerator, Neural_Network, Image_Augmentation
from aucmedi.neural_network.architectures import supported_standardize_mode
from aucmedi.data_processing.subfunctions import Padding
from aucmedi.ensembler import predict_augmenting
# Custom libraries
from retinal_crop import Retinal_Crop

#-----------------------------------------------------#
#                   Configurations                    #
#-----------------------------------------------------#
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Provide path to imaging data
path_riadd = "/storage/riadd2021/Training_Set/"

# Define some parameters
k_fold = 5
processes = 8
batch_queue_size = 16
threads = 12

# Define architectures which should be processed
architectures = ["DenseNet201", "ResNet152", "InceptionV3"]

# Define input shape
input_shape = (224, 224)

#-----------------------------------------------------#
#          AUCMEDI Inference Setup for RIADD          #
#-----------------------------------------------------#
path_images = os.path.join(path_riadd, "Training")
path_csv = os.path.join(path_riadd, "RFMiD_Training_Labels.csv")

# Initialize input data reader
cols = ["DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS", "MS",
        "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST", "AION", "PT",
        "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "OTHER"]
ds = input_interface(interface="csv", path_imagedir=path_images,
                     path_data=path_csv, ohe=True, col_sample="ID",
                     ohe_range=cols)
(index_list, class_ohe, nclasses, class_names, image_format) = ds

# Create result directory
path_res = os.path.join("preds")
if not os.path.exists(path_res) : os.mkdir(path_res)
# Obtain model directory
path_models = os.path.join("models")

# Define Subfunctions
sf_list = [Padding(mode="square"), Retinal_Crop()]

# Initialize Image Augmentation
aug = Image_Augmentation(flip=True, rotate=True, brightness=True, contrast=True,
                         saturation=False, hue=False, scale=False, crop=False,
                         grid_distortion=False, compression=False, gamma=False,
                         gaussian_noise=False, gaussian_blur=False,
                         downscaling=False, elastic_transform=False)

# Iterate over each architecture
for arch in architectures:
    path_arch = os.path.join(path_models, arch)
    # Iterate over each fold of the CV
    for i in range(0, k_fold):
        # Initialize model
        model = Neural_Network(nclasses, channels=3, architecture=arch,
                               workers=processes,
                               batch_queue_size=batch_queue_size,
                               activation_output=activation_output,
                               loss="binary_crossentropy",
                               metrics=["binary_accuracy", AUC(100)],
                               pretrained_weights=True, multiprocessing=True)

        # Obtain standardization mode for current architecture
        sf_standardize = supported_standardize_mode[arch]

        # Load best model
        path_cv_model = os.path.join(path_arch, "cv_" + str(i) + ".model.best.hdf5")
        if os.path.exists(path_cv_model) : model.load(path_cv_model)
        else:
            print("Skipping model:", arch, str(i))

        # Apply Inference Augmenting
        preds = predict_augmenting(model, index_list, path_riadd, n_cycles=5,
                                   img_aug=aug, aggregate="mean",
                                   image_format=image_format, batch_size=32,
                                   resize=input_shape, grayscale=False,
                                   subfunctions=sf_list, seed=None,
                                   standardize_mode=sf_standardize,
                                   workers=threads)

        # Create prediction dataset
        df_index = pd.DataFrame(data={"ID": index_list})
        df_pd = pd.DataFrame(data=preds, columns=["pd_" + s for s in cols])
        df_merged = pd.concat([df_index, df_pd], axis=1, sort=False)
        df_merged.sort_values(by=["ID"], inplace=True)
        # Store predictions to disk
        df_merged.to_csv(os.path.join(path_res, arch + "." + "cv_" + str(i) + \
                                      ".ensemble_train.predictions.csv"),
                         index=False)

        # Garbage collection
        del model




# df_gt = pd.DataFrame(data=class_ohe, columns=["gt_" + s for s in cols])

# load over each model
# make a prediction for complete dataset
# store in ensemble
# train RF/LR/SVM on data to predict final results
# output final results

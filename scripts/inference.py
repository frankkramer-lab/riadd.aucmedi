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
from tensorflow.keras.metrics import AUC
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

# Provide pathes to imaging data
path_riadd = "/storage/riadd2021/Evaluation_Set/"

# Define some parameters
k_fold = 5
processes = 8
batch_queue_size = 16
threads = 16

# Define label columns
cols = ["Disease_Risk", "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM",
        "LS", "MS", "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST",
        "AION", "PT", "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "OTHER"]

# Create result directory
path_res = os.path.join("preds")
if not os.path.exists(path_res) : os.mkdir(path_res)
# Obtain model directory
path_models = os.path.join("models")

#-----------------------------------------------------#
#       General AUCMEDI Inference Pipeline Setup      #
#-----------------------------------------------------#
# Initialize input data reader
ds = input_interface(interface="directory", path_imagedir=path_riadd,
                     path_data=None, training=False)
(index_list, _, _, _, image_format) = ds

# Define Subfunctions
sf_list = [Padding(mode="square"), Retinal_Crop()]

# Initialize Image Augmentation
aug = Image_Augmentation(flip=True, rotate=True, brightness=True, contrast=True,
                         saturation=False, hue=False, scale=False, crop=False,
                         grid_distortion=False, compression=False, gamma=False,
                         gaussian_noise=False, gaussian_blur=False,
                         downscaling=False, elastic_transform=False)

#-----------------------------------------------------#
#            AUCMEDI Classifier Inference             #
#-----------------------------------------------------#
# Define number of classes
nclasses = len(cols[1:])

# Set activation output to sigmoid for multi-label classification
activation_output = "sigmoid"

# Iterate over all classifier architectures
for model_subdir in os.listdir(path_models):
    # Skip all non classifier model subdirs
    if not model_subdir.startswith("classifier_") : continue
    # Identify architecture
    arch = model_subdir.split("_")[1]
    path_arch = os.path.join(path_models, model_subdir)

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

        # Define input shape
        input_shape = model.input_shape[:-1]

        # Initialize Data Generator for prediction
        pred_gen = DataGenerator(index_list, path_riadd, labels=None,
                                 batch_size=64, img_aug=None,
                                 subfunctions=sf_list,
                                 standardize_mode=sf_standardize,
                                 shuffle=False, resize=input_shape,
                                 grayscale=False, prepare_images=False,
                                 sample_weights=None, seed=None,
                                 image_format=image_format, workers=threads)

        # Load best model
        path_cv_model = os.path.join(path_arch, "cv_" + str(i) + ".model.best.hdf5")
        if os.path.exists(path_cv_model) : model.load(path_cv_model)
        else:
            print("Skipping model:", model_subdir, arch, str(i))

        # Use fitted model for predictions
        preds = model.predict(pred_gen)
        # Create prediction dataset
        df_index = pd.DataFrame(data={"ID": index_list})
        df_pd = pd.DataFrame(data=preds, columns=[s for s in cols[1:]])
        df_merged = pd.concat([df_index, df_pd], axis=1, sort=False)
        df_merged.sort_values(by=["ID"], inplace=True)
        # Store predictions to disk
        df_merged.to_csv(os.path.join(path_res, "classifier." + arch + "." + \
                                      "cv_" + str(i) + ".inference.simple.csv"),
                         index=False)

        # Run Inference Augmenting
        preds = predict_augmenting(model, index_list, path_riadd, n_cycles=20,
                                   img_aug=aug, aggregate="mean",
                                   image_format=image_format, batch_size=64,
                                   resize=input_shape, grayscale=False,
                                   subfunctions=sf_list, seed=None,
                                   standardize_mode=sf_standardize,
                                   workers=threads)
        # Create prediction dataset
        df_index = pd.DataFrame(data={"ID": index_list})
        df_pd = pd.DataFrame(data=preds, columns=[s for s in cols[1:]])
        df_merged = pd.concat([df_index, df_pd], axis=1, sort=False)
        df_merged.sort_values(by=["ID"], inplace=True)
        # Store predictions to disk
        df_merged.to_csv(os.path.join(path_res, "classifier." + arch + "." + \
                                      "cv_" + str(i) + ".inference.augmenting.csv"),
                         index=False)

        # Garbage collection
        del pred_gen
        del model

#-----------------------------------------------------#
#              AUCMEDI Detector Inference             #
#-----------------------------------------------------#
# Define number of classes
nclasses = 2

# Set activation output to softmax for binary classification
activation_output = "softmax"

# Iterate over all detector architectures
for model_subdir in os.listdir(path_models):
    # Skip all non detector model subdirs
    if not model_subdir.startswith("detector_") : continue
    # Identify architecture
    arch = model_subdir.split("_")[1]
    path_arch = os.path.join(path_models, model_subdir)

    # Iterate over each fold of the CV
    for i in range(0, k_fold):
        # Initialize model
        model = Neural_Network(nclasses, channels=3, architecture=arch,
                               workers=processes,
                               batch_queue_size=batch_queue_size,
                               activation_output=activation_output,
                               loss="categorical_crossentropy",
                               metrics=["categorical_accuracy", AUC(100)],
                               pretrained_weights=True, multiprocessing=True)

        # Obtain standardization mode for current architecture
        sf_standardize = supported_standardize_mode[arch]

        # Define input shape
        input_shape = model.input_shape[:-1]

        # Initialize Data Generator for prediction
        pred_gen = DataGenerator(index_list, path_riadd, labels=None,
                                 batch_size=64, img_aug=None,
                                 subfunctions=sf_list,
                                 standardize_mode=sf_standardize,
                                 shuffle=False, resize=input_shape,
                                 grayscale=False, prepare_images=False,
                                 sample_weights=None, seed=None,
                                 image_format=image_format, workers=threads)

        # Load best model
        path_cv_model = os.path.join(path_arch, "cv_" + str(i) + ".model.best.hdf5")
        if os.path.exists(path_cv_model) : model.load(path_cv_model)
        else:
            print("Skipping model:", model_subdir, arch, str(i))

        # Use fitted model for predictions
        preds = model.predict(pred_gen)
        # Create prediction dataset
        df_index = pd.DataFrame(data={"ID": index_list})
        df_pd = pd.DataFrame(data={cols[0]: preds[:, 1]})
        df_merged = pd.concat([df_index, df_pd], axis=1, sort=False)
        df_merged.sort_values(by=["ID"], inplace=True)
        # Store predictions to disk
        df_merged.to_csv(os.path.join(path_res, "detector." + arch + "." + \
                                      "cv_" + str(i) + ".inference.simple.csv"),
                         index=False)

        # Run Inference Augmenting
        preds = predict_augmenting(model, index_list, path_images, n_cycles=20,
                                   img_aug=aug, aggregate="mean",
                                   image_format=image_format, batch_size=64,
                                   resize=input_shape, grayscale=False,
                                   subfunctions=sf_list, seed=None,
                                   standardize_mode=sf_standardize,
                                   workers=threads)
        # Create prediction dataset
        df_index = pd.DataFrame(data={"ID": index_list})
        df_pd = pd.DataFrame(data={cols[0]: preds[:, 1]})
        df_merged = pd.concat([df_index, df_pd], axis=1, sort=False)
        df_merged.sort_values(by=["ID"], inplace=True)
        # Store predictions to disk
        df_merged.to_csv(os.path.join(path_res, "detector." + arch + "." + \
                                      "cv_" + str(i) + ".inference.augmenting.csv"),
                         index=False)

        # Garbage collection
        del pred_gen
        del model

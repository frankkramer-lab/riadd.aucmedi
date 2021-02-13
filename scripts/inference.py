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
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, \
                                       ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.metrics import AUC
# AUCMEDI libraries
from aucmedi import input_interface, DataGenerator, Neural_Network, Image_Augmentation
from aucmedi.neural_network.architectures import supported_standardize_mode
from aucmedi.utils.class_weights import compute_sample_weights
from aucmedi.data_processing.subfunctions import Padding
from aucmedi.sampling import sampling_kfold

#-----------------------------------------------------#
#                   Configurations                    #
#-----------------------------------------------------#
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Provide pathes to imaging data
path_riadd = "/storage/riadd2021/Training_Set/"

# Define some parameters
k_fold = 5
processes = 50
batch_queue_size = 75
threads = 8

# Define architectures which should be processed
architectures = ["VGG16", "DenseNet169", "ResNet101", "ResNet152", "EfficientNetB4",
                 "ResNeXt101", "InceptionResNetV2"]

#-----------------------------------------------------#
#          AUCMEDI Classifier Setup for RIADD         #
#-----------------------------------------------------#
# Initialize input data reader
cols = ["DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS", "MS", "CSR",
        "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST", "AION", "PT", "RT", "RS",
        "CRS", "EDN", "RPEC", "MHL", "RP", "OTHER"]
ds = input_interface(interface="directory", path_imagedir=path_riadd,
                     path_data=None, training=False)
(index_list, class_ohe, nclasses, class_names, image_format) = ds
nclasses = len(cols)

# Create result directory
path_res = os.path.join("preds")
if not os.path.exists(path_res) : os.mkdir(path_res)
# Obtain model directory
path_models = os.path.join("models")

# Define Subfunctions
sf_list = [Padding(mode="square")]

# Set activation output to sigmoid for multi-label classification
activation_output = "sigmoid"

#-----------------------------------------------------#
#        AUCMEDI Classifier Inference for RIADD       #
#-----------------------------------------------------#
# Run a k-fold CV for each architecture
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
        # Obtain standard input shape for current architecture
        input_shape = model.input_shape[:-1]

        # Initialize Data Generator for prediction
        pred_gen = DataGenerator(index_list, path_riadd, labels=None,
                                 batch_size=32, img_aug=None,
                                 subfunctions=sf_list,
                                 standardize_mode=sf_standardize,
                                 shuffle=False, resize=input_shape,
                                 grayscale=False, prepare_images=False,
                                 sample_weights=None, seed=None,
                                 image_format=image_format, workers=8)

        # Load best model
        path_cv_model = os.path.join(path_arch, "cv_" + str(i) + ".model.best.hdf5")
        model.load(path_cv_model)

        # Use fitted model for predictions
        preds = model.predict(test_gen)
        # Create prediction dataset
        df_index = pd.DataFrame(data={"ID": index_list})
        df_dr = pd.DataFrame(data={"Disease_Risk": [0] * len(index_list)})
        df_pd = pd.DataFrame(data=preds, columns=[s for s in cols])
        df_merged = pd.concat([df_index, df_dr, df_pd], axis=1, sort=False)
        df_merged.sort_values(by=["ID"], inplace=True)
        # Store predictions to disk
        df_merged.to_csv(os.path.join(path_res, arch + "." + "cv_" + str(i) + \
                                      ".simple.predictions.csv"),
                         index=False)

        # Apply Inference Augmenting
        preds = predict_augmenting(model, index_list, path_riadd, n_cycles=20,
                                   img_aug=None, aggregate="mean",
                                   image_format=image_format, batch_size=32,
                                   resize=input_shape, grayscale=False,
                                   subfunctions=sf_list, seed=None,
                                   standardize_mode=sf_standardize, workers=8)
        # Create prediction dataset
        df_index = pd.DataFrame(data={"ID": index_list})
        df_dr = pd.DataFrame(data={"Disease_Risk": [0] * len(index_list)})
        df_pd = pd.DataFrame(data=preds, columns=[s for s in cols])
        df_merged = pd.concat([df_index, df_dr, df_pd], axis=1, sort=False)
        df_merged.sort_values(by=["ID"], inplace=True)
        # Store predictions to disk
        df_merged.to_csv(os.path.join(path_res, arch + "." + "cv_" + str(i) + \
                                      ".augmenting_mean.predictions.csv"),
                         index=False)

        # Apply Inference Augmenting
        preds = predict_augmenting(model, index_list, path_riadd, n_cycles=20,
                                   img_aug=None, aggregate="softmax",
                                   image_format=image_format, batch_size=32,
                                   resize=input_shape, grayscale=False,
                                   subfunctions=sf_list, seed=None,
                                   standardize_mode=sf_standardize, workers=8)
        # Create prediction dataset
        df_index = pd.DataFrame(data={"ID": index_list})
        df_dr = pd.DataFrame(data={"Disease_Risk": [0] * len(index_list)})
        df_pd = pd.DataFrame(data=preds, columns=[s for s in cols])
        df_merged = pd.concat([df_index, df_dr, df_pd], axis=1, sort=False)
        df_merged.sort_values(by=["ID"], inplace=True)
        # Store predictions to disk
        df_merged.to_csv(os.path.join(path_res, arch + "." + "cv_" + str(i) + \
                                      ".augmenting_softmax.predictions.csv"),
                         index=False)

        # Garbage collection
        del pred_gen
        del model

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

# Provide pathes to imaging and annotation data
path_riadd = "/storage/riadd2021/Training_Set/"

# Define some parameters
k_fold = 5
processes = 50
batch_queue_size = 75
threads = 8

# Define architectures which should be processed
architectures = ["VGG16", "DenseNet169", "ResNet152", "EfficientNetB4", "Xception",
                 "ResNeXt101", "InceptionResNetV2"]

#-----------------------------------------------------#
#          AUCMEDI Classifier Setup for RIADD         #
#-----------------------------------------------------#
path_images = os.path.join(path_riadd, "Training")
path_csv = os.path.join(path_riadd, "RFMiD_Training_Labels.csv")

# Initialize input data reader
cols = ["DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS", "MS", "CSR",
        "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST", "AION", "PT", "RT", "RS",
        "CRS", "EDN", "RPEC", "MHL", "RP", "OTHER"]
ds = input_interface(interface="csv", path_imagedir=path_images,
                     path_data=path_csv, ohe=True, col_sample="ID",
                     ohe_range=cols)
(index_list, class_ohe, nclasses, class_names, image_format) = ds

# Create models directory
path_models = os.path.join("models")
if not os.path.exists(path_models) : os.mkdir(path_models)

# Sample dataset via k-fold cross-validation
subsets = sampling_kfold(index_list, class_ohe, n_splits=k_fold,
                         stratified=True, iterative=True, seed=0)

# Store sampling to disk
sampling_dict = {}
for i, fold in enumerate(subsets):
    (x_train, y_train, x_val, y_val) = fold
    sampling_dict["cv_" + str(i)] = {"train": x_train.tolist(),
                                     "val": x_val.tolist()}
with open(os.path.join(path_models, "sampling.json"), "w") as file:
    json.dump(sampling_dict, file, indent=2)

# Compute sample weights
sample_weights = compute_sample_weights(ohe_array=y_train)
# Initialize Image Augmentation
aug = Image_Augmentation(flip=True, rotate=True, brightness=True, contrast=True,
                         saturation=False, hue=False, scale=True, crop=False,
                         grid_distortion=True, compression=False, gamma=False,
                         gaussian_noise=False, gaussian_blur=False,
                         downscaling=False, elastic_transform=False)
# Define Subfunctions
sf_list = [Padding(mode="square")]
# Set activation output to sigmoid for multi-label classification
activation_output = "sigmoid"

#-----------------------------------------------------#
#        AUCMEDI Classifier Training for RIADD        #
#-----------------------------------------------------#
# Run a k-fold CV for each architecture
for arch in architectures:
    # Create architecture directory
    path_arch = os.path.join(path_models, arch)
    if not os.path.exists(path_arch) : os.mkdir(path_arch)

    # Iterate over each fold of the CV
    for i, fold in enumerate(subsets):
        # Obtain data samplings
        (x_train, y_train, x_val, y_val) = fold

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

        # Initialize training and validation Data Generators
        train_gen = DataGenerator(x_train, path_images, labels=y_train,
                                  batch_size=32, img_aug=aug, shuffle=True,
                                  subfunctions=sf_list, resize=input_shape,
                                  standardize_mode=sf_standardize,
                                  grayscale=False, prepare_images=False,
                                  sample_weights=sample_weights, seed=None,
                                  image_format=image_format, workers=threads)
        val_gen = DataGenerator(x_val, path_images, labels=y_val, batch_size=32,
                                img_aug=None, subfunctions=sf_list, shuffle=False,
                                standardize_mode=sf_standardize, resize=input_shape,
                                grayscale=False, prepare_images=False, seed=None,
                                sample_weights=None, image_format=image_format,
                                workers=8)

        # Define callbacks
        cb_mc = ModelCheckpoint(os.path.join(path_arch, "cv_" + str(i) + \
                                             ".model.best.hdf5"),
                                monitor="val_loss", verbose=1,
                                save_best_only=True, mode="min")
        cb_cl = CSVLogger(os.path.join(path_arch, "cv_" + str(i) + ".logs.csv"),
                          separator=',')
        cb_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8,
                                  verbose=1, mode='min', min_lr=1e-6)
        cb_es = EarlyStopping(monitor='val_loss', patience=16, verbose=1)
        callbacks = [cb_mc, cb_cl, cb_lr, cb_es]

        # Train model
        model.train(train_gen, val_gen, epochs=150, iterations=150,
                    callbacks=callbacks, transfer_learning=True)

        # Dump latest model
        model.dump(os.path.join(path_arch, "cv_" + str(i) + ".model.last.hdf5"))

        # Garbage collection
        del train_gen
        del val_gen
        del model
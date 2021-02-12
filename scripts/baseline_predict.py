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
from aucmedi.data_processing.subfunctions import Padding
from aucmedi.ensembler import predict_augmenting

#-----------------------------------------------------#
#              AUCMEDI Baseline for RIADD             #
#-----------------------------------------------------#
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Provide pathes to imaging and annotation data
path_test = "/home/mudomini/data/RIADD/Training_Set/Evaluation_Set"
# path_test = "/storage/riadd2021/Evaluation_Set"

# Initialize input data reader
cols = ["DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS", "MS", "CSR",
        "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST", "AION", "PT", "RT", "RS",
        "CRS", "EDN", "RPEC", "MHL", "RP", "OTHER"]
ds = input_interface(interface="directory", path_imagedir=path_test,
                     path_data=None, training=False)
(index_list, class_ohe, nclasses, class_names, image_format) = ds
nclasses = len(cols)

# Create result directory
path_res = os.path.join("preds")
if not os.path.exists(path_res) : os.mkdir(path_res)

# Define Subfunctions
sf_list = [Padding(mode="square")]

# Set activation output to sigmoid for multi-label classification
activation_output = "sigmoid"
# Define architectures which should be processed
architectures = ["VGG16", "DenseNet169", "ResNet101", "ResNet152", "EfficientNetB4",
                 "Xception", "ResNeXt101", "InceptionResNetV2"]

# Create pipelines for each architectures
path_model = os.path.join("results")
for arch in architectures:
    # Initialize model
    model = Neural_Network(nclasses, channels=3, architecture=arch,
                           workers=50, batch_queue_size=75,
                           activation_output=activation_output,
                           loss="binary_crossentropy",
                           metrics=["binary_accuracy", AUC(100)],
                           pretrained_weights=True, multiprocessing=True)
    model.model.summary()

    # Obtain standardization mode for current architecture
    sf_standardize = supported_standardize_mode[arch]
    # Obtain standard input shape for current architecture
    input_shape = model.input_shape[:-1]

    # Load best model
    model.load(os.path.join(path_model, arch + ".model.best.hdf5"))

    # Initialize testing Data Generator
    test_gen = DataGenerator(index_list, path_test, labels=None, batch_size=32,
                             img_aug=None, subfunctions=sf_list, standardize_mode=sf_standardize,
                             shuffle=False, resize=input_shape, grayscale=False, prepare_images=False,
                             sample_weights=None, seed=None, image_format=image_format, workers=8)

    # Use fitted model for predictions
    preds = model.predict(test_gen)
    # Create prediction dataset
    df_index = pd.DataFrame(data={"index:": index_list})
    df_pd = pd.DataFrame(data=preds, columns=[s for s in cols])
    df_merged = pd.concat([df_index, df_pd], axis=1, sort=False)
    # Store predictions to disk
    df_merged.to_csv(os.path.join(path_res, arch + ".simple.predictions.csv"),
                     index=False)

    # Apply Inference Augmenting
    preds = predict_augmenting(model, index_list, path_test, n_cycles=20,
                               img_aug=None, aggregate="mean",
                               image_format=image_format, batch_size=32,
                               resize=input_shape, grayscale=False,
                               subfunctions=sf_list, seed=None,
                               standardize_mode=sf_standardize, workers=8)
    # Create prediction dataset
    df_index = pd.DataFrame(data={"index:": index_list})
    df_pd = pd.DataFrame(data=preds, columns=[s for s in cols])
    df_merged = pd.concat([df_index, df_pd], axis=1, sort=False)
    # Store predictions to disk
    df_merged.to_csv(os.path.join(path_res, arch + ".augmenting_mean.predictions.csv"),
                     index=False)

    # Apply Inference Augmenting
    preds = predict_augmenting(model, index_list, path_test, n_cycles=20,
                               img_aug=None, aggregate="softmax",
                               image_format=image_format, batch_size=32,
                               resize=input_shape, grayscale=False,
                               subfunctions=sf_list, seed=None,
                               standardize_mode=sf_standardize, workers=8)
    # Create prediction dataset
    df_index = pd.DataFrame(data={"index:": index_list})
    df_pd = pd.DataFrame(data=preds, columns=[s for s in cols])
    df_merged = pd.concat([df_index, df_pd], axis=1, sort=False)
    # Store predictions to disk
    df_merged.to_csv(os.path.join(path_res, arch + ".augmenting_softmax.predictions.csv"),
                     index=False)

    # Garbage collection
    del test_gen
    del model

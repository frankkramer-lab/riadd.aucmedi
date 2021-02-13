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
from aucmedi.sampling import sampling_split
from aucmedi.ensembler import predict_augmenting
from retinal_crop import Retinal_Crop

#-----------------------------------------------------#
#              AUCMEDI Baseline for RIADD             #
#-----------------------------------------------------#
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Provide pathes to imaging and annotation data
path_riadd = "/storage/riadd2021/Training_Set/"
path_images = os.path.join(path_riadd, "Training")
path_csv = os.path.join(path_riadd, "RFMiD_Training_Labels.csv")

# Initialize input data reader
cols = ["DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS", "MS", "CSR",
        "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST", "AION", "PT", "RT", "RS",
        "CRS", "EDN", "RPEC", "MHL", "RP", "OTHER"]
ds = input_interface(interface="csv", path_imagedir=path_images, path_data=path_csv,
                     ohe=True, col_sample="ID", ohe_range=cols)
(index_list, class_ohe, nclasses, class_names, image_format) = ds

# Create result directory
path_res = os.path.join("results")
if not os.path.exists(path_res) : os.mkdir(path_res)

# Sample dataset
subsets = sampling_split(index_list, class_ohe, sampling=[0.9, 0.1],
                         stratified=True, iterative=True, seed=0)
X_train = subsets[0][0]
y_train = subsets[0][1]
X_test = subsets[1][0]
y_test = subsets[1][1]

# Store sampling to disk
with open(os.path.join(path_res, "sampling.json"), "w") as file:
    json_dict = {"train": X_train.tolist(),
                 "test": X_test.tolist()}
    json.dump(json_dict, file, indent=2)

# Compute sample weights
sample_weights = compute_sample_weights(ohe_array=y_train)

# Initialize Image Augmentation
aug = Image_Augmentation(flip=True, rotate=True, brightness=True, contrast=True,
                         saturation=True, hue=True, scale=True, crop=False,
                         grid_distortion=False, compression=False,
                         gaussian_noise=False, gaussian_blur=False,
                         downscaling=False, gamma=False,
                         elastic_transform=False)

# Define Subfunctions
sf_list = [Padding(mode="square"), Retinal_Crop]

# Set activation output to sigmoid for multi-label classification
activation_output = "sigmoid"
# Define architectures which should be processed
architectures = ["VGG16", "DenseNet169", "ResNet101", "ResNet152", "EfficientNetB4",
                 "Xception", "ResNeXt101", "InceptionResNetV2"]

# Create pipelines for each architectures
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

    # Initialize training and validation Data Generators
    train_gen = DataGenerator(X_train, path_images, labels=y_train, batch_size=32,
                             img_aug=aug, subfunctions=sf_list, standardize_mode=sf_standardize,
                             shuffle=True, resize=input_shape, grayscale=False, prepare_images=False,
                             sample_weights=sample_weights, seed=None, image_format=image_format, workers=8)
    val_gen = DataGenerator(X_test, path_images, labels=y_test, batch_size=32,
                            img_aug=aug, subfunctions=sf_list, standardize_mode=sf_standardize,
                            shuffle=False, resize=input_shape, grayscale=False, prepare_images=False,
                            sample_weights=None, seed=None, image_format=image_format, workers=8)

    # Define callbacks
    cb_mc = ModelCheckpoint(os.path.join(path_res, arch + ".model.best.hdf5"),
                            monitor="val_loss", verbose=1,
                            save_best_only=True, mode="min")
    cb_cl = CSVLogger(os.path.join(path_res, arch + ".logs.csv"), separator=',')
    cb_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=8,
                              verbose=1, mode='min', min_lr=1e-6)
    cb_es = EarlyStopping(monitor='val_loss', patience=16, verbose=1)
    callbacks = [cb_mc, cb_cl, cb_lr, cb_es]

    # Train model
    model.train(train_gen, val_gen, epochs=150, iterations=150, callbacks=callbacks,
                transfer_learning=True)

    # Load best model
    model.load(os.path.join(path_res, arch + ".model.best.hdf5"))

    # Initialize testing Data Generator
    test_gen = DataGenerator(X_test, path_images, labels=None, batch_size=32,
                             img_aug=None, subfunctions=sf_list, standardize_mode=sf_standardize,
                             shuffle=False, resize=input_shape, grayscale=False, prepare_images=False,
                             sample_weights=None, seed=None, image_format=image_format, workers=8)

    # Use fitted model for predictions
    preds = model.predict(test_gen)
    # Create prediction dataset
    df_index = pd.DataFrame(data={"index:": X_test})
    df_gt = pd.DataFrame(data=y_test, columns=["gt_" + s for s in cols])
    df_pd = pd.DataFrame(data=preds, columns=["pd_" + s for s in cols])
    df_merged = pd.concat([df_index, df_gt, df_pd], axis=1, sort=False)
    # Store predictions to disk
    df_merged.to_csv(os.path.join(path_res, arch + ".simple.predictions.csv"),
                     index=False)

    # Apply Inference Augmenting
    preds = predict_augmenting(model, X_test, path_images, n_cycles=20,
                               img_aug=aug, aggregate="mean",
                               image_format=image_format, batch_size=32,
                               resize=input_shape, grayscale=False,
                               subfunctions=sf_list, seed=None,
                               standardize_mode=sf_standardize, workers=8)
    # Create prediction dataset
    df_index = pd.DataFrame(data={"index:": X_test})
    df_gt = pd.DataFrame(data=y_test, columns=["gt_" + s for s in cols])
    df_pd = pd.DataFrame(data=preds, columns=["pd_" + s for s in cols])
    df_merged = pd.concat([df_index, df_gt, df_pd], axis=1, sort=False)
    # Store predictions to disk
    df_merged.to_csv(os.path.join(path_res, arch + ".augmenting_mean.predictions.csv"),
                     index=False)

    # Apply Inference Augmenting
    preds = predict_augmenting(model, X_test, path_images, n_cycles=20,
                               img_aug=aug, aggregate="softmax",
                               image_format=image_format, batch_size=32,
                               resize=input_shape, grayscale=False,
                               subfunctions=sf_list, seed=None,
                               standardize_mode=sf_standardize, workers=8)
    # Create prediction dataset
    df_index = pd.DataFrame(data={"index:": X_test})
    df_gt = pd.DataFrame(data=y_test, columns=["gt_" + s for s in cols])
    df_pd = pd.DataFrame(data=preds, columns=["pd_" + s for s in cols])
    df_merged = pd.concat([df_index, df_gt, df_pd], axis=1, sort=False)
    # Store predictions to disk
    df_merged.to_csv(os.path.join(path_res, arch + ".augmenting_softmax.predictions.csv"),
                     index=False)

    # Garbage collection
    del train_gen
    del val_gen
    del test_gen
    del model

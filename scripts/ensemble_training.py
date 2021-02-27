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
import pickle
from tensorflow.keras.metrics import AUC
# Sklearn libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# AUCMEDI libraries
from aucmedi import input_interface, DataGenerator, Neural_Network, Image_Augmentation
from aucmedi.neural_network.architectures import supported_standardize_mode
from aucmedi.data_processing.subfunctions import Padding
from aucmedi.ensembler import predict_augmenting
# Custom libraries
from retinal_crop import Retinal_Crop

#-----------------------------------------------------#
#              Configurations and Setup               #
#-----------------------------------------------------#
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# Provide path to imaging data
path_riadd = "/storage/riadd2021/Upsampled_Set/"

path_images = os.path.join(path_riadd, "images")
path_csv = os.path.join(path_riadd, "data.csv")

# Define some parameters
k_fold = 5
processes = 8
batch_queue_size = 16
threads = 32

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
# Initialize input data reader
ds = input_interface(interface="csv", path_imagedir=path_images,
                     path_data=path_csv, ohe=True, col_sample="ID",
                     ohe_range=cols[1:])
(index_list, class_ohe, nclasses, class_names, image_format) = ds

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
        # Skip
        if os.path.exists(os.path.join(path_res, "classifier." + arch + "." + \
                                      "cv_" + str(i) + ".ensemble_train.csv")):
            continue
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

        # Load best model
        path_cv_model = os.path.join(path_arch, "cv_" + str(i) + ".model.best.hdf5")
        if os.path.exists(path_cv_model) : model.load(path_cv_model)
        else:
            print("Skipping model:", model_subdir, arch, str(i))

        # Apply Inference Augmenting
        preds = predict_augmenting(model, index_list, path_images, n_cycles=5,
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
                                      "cv_" + str(i) + ".ensemble_train.csv"),
                         index=False)

        # Garbage collection
        del model

#-----------------------------------------------------#
#              AUCMEDI Detector Inference             #
#-----------------------------------------------------#
# Initialize input data reader
ds = input_interface(interface="csv", path_imagedir=path_images,
                     path_data=path_csv, ohe=False, col_sample="ID",
                     col_class=cols[0])
(index_list, class_ohe, nclasses, class_names, image_format) = ds

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
        # Skip
        if os.path.exists(os.path.join(path_res, "detector." + arch + "." + \
                                      "cv_" + str(i) + ".ensemble_train.csv")):
            continue
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

        # Load best model
        path_cv_model = os.path.join(path_arch, "cv_" + str(i) + ".model.best.hdf5")
        if os.path.exists(path_cv_model) : model.load(path_cv_model)
        else:
            print("Skipping model:", model_subdir, arch, str(i))

        # Apply Inference Augmenting
        preds = predict_augmenting(model, index_list, path_images, n_cycles=5,
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
                                      "cv_" + str(i) + ".ensemble_train.csv"),
                         index=False)

        # Garbage collection
        del model

#-----------------------------------------------------#
#          AUCMEDI Boost CRS & EDN Inference          #
#-----------------------------------------------------#
# Iterate over all detector architectures
for model_subdir in os.listdir(path_models):
    # Skip all non detector model subdirs
    if not model_subdir.startswith("boostCRS_") or not model_subdir.startswith("boostEDN_"): continue
    # Identify boost class
    boostclass = model_subdir.split("_")[0]
    # Identify class
    boosted_class = boostclass[-3:]

    # Initialize input data reader
    ds = input_interface(interface="csv", path_imagedir=path_images,
                         path_data=path_csv, ohe=False, col_sample="ID",
                         col_class=boosted_class)
    (index_list, class_ohe, nclasses, class_names, image_format) = ds

    # Set activation output to softmax for binary classification
    activation_output = "softmax"

    # Identify architecture
    arch = model_subdir.split("_")[1]
    path_arch = os.path.join(path_models, model_subdir)

    # Iterate over each fold of the CV
    for i in range(0, 3):
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

        # Load best model
        path_cv_model = os.path.join(path_arch, "cv_" + str(i) + ".model.best.hdf5")
        if os.path.exists(path_cv_model) : model.load(path_cv_model)
        else:
            print("Skipping model:", model_subdir, arch, str(i))

        # Apply Inference Augmenting
        preds = predict_augmenting(model, index_list, path_images, n_cycles=5,
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
        df_merged.to_csv(os.path.join(path_res, boostclass + "." + arch + "." + \
                                      "cv_" + str(i) + ".ensemble_train.csv"),
                         index=False)

        # Garbage collection
        del model

#-----------------------------------------------------#
#           AUCMEDI Boost Cropping Inference          #
#-----------------------------------------------------#
# Initialize input data reader
ds = input_interface(interface="csv", path_imagedir=path_images,
                     path_data=path_csv, ohe=True, col_sample="ID",
                     ohe_range=cols[1:])
(index_list, class_ohe, nclasses, class_names, image_format) = ds

# Set activation output to sigmoid for multi-label classification
activation_output = "sigmoid"

# Define input shape
resize_shape = (416, 416)
input_shape = (224, 224)
# Define Subfunctions
sf_list = [Padding(mode="square"), Retinal_Crop(), Resize(resize_shape),
           Crop(input_shape)]

# Iterate over all classifier architectures
for model_subdir in os.listdir(path_models):
    # Skip all non classifier model subdirs
    if not model_subdir.startswith("boostCROP_") : continue
    # Identify architecture
    arch = model_subdir.split("_")[1]
    path_arch = os.path.join(path_models, model_subdir)

    # Iterate over each fold of the CV
    for i in range(0, 3):
        # Initialize architecture
        nn_arch = architecture_dict[arch](channels=3, input_shape=input_shape)

        # Initialize model
        model = Neural_Network(nclasses, channels=3, architecture=nn_arch,
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
            print("Skipping model:", model_subdir, arch, str(i))

        # Apply Inference Augmenting
        preds = predict_augmenting(model, index_list, path_images, n_cycles=5,
                                   img_aug=aug, aggregate="mean",
                                   image_format=image_format, batch_size=64,
                                   resize=None, grayscale=False,
                                   subfunctions=sf_list, seed=None,
                                   standardize_mode=sf_standardize,
                                   workers=threads)

        # Create prediction dataset
        df_index = pd.DataFrame(data={"ID": index_list})
        df_pd = pd.DataFrame(data=preds, columns=[s for s in cols[1:]])
        df_merged = pd.concat([df_index, df_pd], axis=1, sort=False)
        df_merged.sort_values(by=["ID"], inplace=True)
        # Store predictions to disk
        df_merged.to_csv(os.path.join(path_res, "boostCROP." + arch + "." + \
                                      "cv_" + str(i) + ".ensemble_train.csv"),
                         index=False)

        # Garbage collection
        del model

#-----------------------------------------------------#
#                  Ensemble Training                  #
#-----------------------------------------------------#
# Iterate over all ensemble learning predictions
dt_pred = None
for pred_file in sorted(os.listdir(path_res)):
    if not pred_file.split(".")[3] == "ensemble_train" : continue
    # Load label prediction
    pred = pd.read_csv(os.path.join(path_res, pred_file), sep=",", header=0)
    # Obtain column prefix
    prefix = ".".join(pred_file.split(".")[0:3])
    # Rename columns
    label_cols = list(pred.columns[1:])
    label_cols = [prefix + "." + label for label in label_cols]
    pred.columns = ["ID"] + label_cols
    # Merge predictions
    if dt_pred is None : dt_pred = pred
    else : dt_pred = dt_pred.merge(pred, on="ID")

# Load ground truth for all classes
ds = input_interface(interface="csv", path_imagedir=path_images,
                     path_data=path_csv, ohe=True, col_sample="ID",
                     ohe_range=cols)
(index_list, class_ohe, _, _, _) = ds

# Obtain features and labels for ML model
df_index = pd.DataFrame(data={"ID": index_list})
dt_gt = pd.DataFrame(class_ohe, columns=["gt_" + s for s in cols])
df_merged = pd.concat([df_index, dt_gt], axis=1, sort=False)
data = dt_pred.merge(df_merged, on="ID")

features = data.drop(["gt_" + s for s in cols] + ["ID"], axis=1).to_numpy()
labels = data[["gt_" + s for s in cols]].to_numpy()

# Create ensemble model subdirectory
path_mldir = os.path.join(path_models, "ensemble")
if not os.path.exists(path_mldir) : os.mkdir(path_mldir)

# Iterate over each class
for i, c in enumerate(cols):
    # Obtain ground truth labels for class c
    class_label = labels[:,i]

    # Train Logistic Regression model
    print("Training Logistic Regression model for:", c)
    lr_model = LogisticRegression(class_weight="balanced", max_iter=1000)
    lr_model.fit(features, class_label)
    # Store LR model to disk
    path_lrmodel = os.path.join(path_mldir, "model_lr." + c + ".pickle")
    with open(path_lrmodel, "wb") as pickle_writer:
        pickle.dump(lr_model, pickle_writer, protocol=pickle.HIGHEST_PROTOCOL)

    # Train Random Forest model
    print("Training Random Forest model for:", c)
    rf_model = RandomForestClassifier(class_weight="balanced", n_estimators=150)
    rf_model.fit(features, class_label)
    # Store RF model to disk
    path_rfmodel = os.path.join(path_mldir, "model_rf." + c + ".pickle")
    with open(path_rfmodel, "wb") as pickle_writer:
        pickle.dump(rf_model, pickle_writer, protocol=pickle.HIGHEST_PROTOCOL)

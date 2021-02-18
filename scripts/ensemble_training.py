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
# path_riadd = "/storage/riadd2021/Upsampled_Set/"
path_riadd = "/home/mudomini/data/RIADD/Upsampled_Set/"

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
path_images = os.path.join(path_riadd, "images")
path_csv = os.path.join(path_riadd, "data.csv")

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

# # Define Subfunctions
# sf_list = [Padding(mode="square"), Retinal_Crop()]
#
# # Set activation output to sigmoid for multi-label classification
# activation_output = "sigmoid"
#
# # Initialize Image Augmentation
# aug = Image_Augmentation(flip=True, rotate=True, brightness=True, contrast=True,
#                          saturation=False, hue=False, scale=False, crop=False,
#                          grid_distortion=False, compression=False, gamma=False,
#                          gaussian_noise=False, gaussian_blur=False,
#                          downscaling=False, elastic_transform=False)
#
# # Iterate over each architecture
# for arch in architectures:
#     path_arch = os.path.join(path_models, arch)
#     # Iterate over each fold of the CV
#     for i in range(0, k_fold):
#         # Initialize model
#         model = Neural_Network(nclasses, channels=3, architecture=arch,
#                                workers=processes,
#                                batch_queue_size=batch_queue_size,
#                                activation_output=activation_output,
#                                loss="binary_crossentropy",
#                                metrics=["binary_accuracy", AUC(100)],
#                                pretrained_weights=True, multiprocessing=True)
#
#         # Obtain standardization mode for current architecture
#         sf_standardize = supported_standardize_mode[arch]
#
#         # Load best model
#         path_cv_model = os.path.join(path_arch, "cv_" + str(i) + ".model.best.hdf5")
#         if os.path.exists(path_cv_model) : model.load(path_cv_model)
#         else:
#             print("Skipping model:", arch, str(i))
#
#         # Apply Inference Augmenting
#         preds = predict_augmenting(model, index_list, path_images, n_cycles=5,
#                                    img_aug=aug, aggregate="mean",
#                                    image_format=image_format, batch_size=64,
#                                    resize=input_shape, grayscale=False,
#                                    subfunctions=sf_list, seed=None,
#                                    standardize_mode=sf_standardize,
#                                    workers=threads)
#
#         # Create prediction dataset
#         df_index = pd.DataFrame(data={"ID": index_list})
#         df_pd = pd.DataFrame(data=preds, columns=[s for s in cols])
#         df_merged = pd.concat([df_index, df_pd], axis=1, sort=False)
#         df_merged.sort_values(by=["ID"], inplace=True)
#         # Store predictions to disk
#         df_merged.to_csv(os.path.join(path_res, arch + "." + "cv_" + str(i) + \
#                                       ".ensemble_train.predictions.csv"),
#                          index=False)
#
#         # Garbage collection
#         del model

#-----------------------------------------------------#
#         AUCMEDI Ensemble Training for RIADD         #
#-----------------------------------------------------#
# Iterate over all ensemble learning preidctions
dt_pred = None
for pred_file in sorted(os.listdir(path_res)):
    if not pred_file.split(".")[2] == "ensemble_train" : continue
    # Load label prediction
    pred = pd.read_csv(os.path.join(path_res, pred_file), sep=",", header=0)
    # Rename columns
    prefix = ".".join(pred_file.split(".")[0:2])
    label_cols = list(pred.columns[1:])
    label_cols = [prefix + "." + label.lstrip("pd_") for label in label_cols] # Remove lstrip later if predictions are re-computed
    pred.columns = ["ID"] + label_cols
    # Merge predictions
    if dt_pred is None : dt_pred = pred
    else : dt_pred = dt_pred.merge(pred, on="ID")

# Obtain features and labels for ML model
df_index = pd.DataFrame(data={"ID": index_list})
dt_gt = pd.DataFrame(class_ohe, columns=["gt_" + s for s in cols])
df_merged = pd.concat([df_index, dt_gt], axis=1, sort=False)
data = dt_pred.merge(df_merged, on="ID")

features = data.drop(["gt_" + s for s in cols] + ["ID"], axis=1).to_numpy()
labels = data[["gt_" + s for s in cols]].to_numpy()

# Train Random Forest model
ml_model = RandomForestClassifier()
ml_model.fit(features, labels)

# Store model to disk
path_mldir = os.path.join(path_models, "ensemble")
if not os.path.exists(path_mldir) : os.mkdir(path_mldir)

with open(os.path.join(path_mldir, "labels.rf.pickle"), "wb") as pickle_writer:
    pickle.dump(ml_model, pickle_writer, protocol=pickle.HIGHEST_PROTOCOL)

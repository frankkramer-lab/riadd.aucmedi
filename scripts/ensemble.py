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

#-----------------------------------------------------#
#                   Configurations                    #
#-----------------------------------------------------#
# Define mode of predictions to utilize
# ["simple", "augmenting"]
mode = "simple"

# Provide pathes to prediction data
path_preds = "preds"

# Provide path to ensemble model directory
path_modeldir = os.path.join("models", "ensemble")

# Define label columns
cols = ["Disease_Risk", "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM",
        "LS", "MS", "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST",
        "AION", "PT", "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "OTHER"]

# #-----------------------------------------------------#
# #            Ensemble Learning: Averaging             #
# #-----------------------------------------------------#
# # Iterate over all label preidctions
# dt_pred = pd.DataFrame([])
# for pred_file in os.listdir(path_preds):
#     if not pred_file.split(".")[2] == "labels" : continue
#     # Load label prediction
#     pred = pd.read_csv(os.path.join(path_preds, pred_file), sep=",", header=0)
#     # Add to list
#     dt_pred = dt_pred.append(pred)
#
#
# # Compute mean
# dt_pred = dt_pred.groupby("ID").mean()
#

#-----------------------------------------------------#
#             Ensemble Learning: Stacking             #
#-----------------------------------------------------#
# Iterate over the predictions from all models (detector & classifier)
dt_pred = None
for pred_file in sorted(os.listdir(path_preds)):
    if not pred_file.split(".")[3] == "inference" : continue
    if not pred_file.split(".")[4] == mode : continue
    # Load label prediction
    pred = pd.read_csv(os.path.join(path_preds, pred_file), sep=",", header=0)
    # Rename columns
    prefix = ".".join(pred_file.split(".")[0:3])
    label_cols = list(pred.columns[1:])
    label_cols = [prefix + "." + label for label in label_cols]
    pred.columns = ["ID"] + label_cols
    # Merge predictions
    if dt_pred is None : dt_pred = pred
    else : dt_pred = dt_pred.merge(pred, on="ID")

# Obtain features table
features = dt_pred.drop("ID", axis=1).to_numpy()

# Apply Logistic Regression and Random Forest Inference
for ml in ["lr", "rf"]:
    # Iterate over each class
    preds_list = []
    for c in cols:
        # Load machine learning model
        path_model = os.path.join(path_modeldir,
                                  "model_" + ml + "." + c + ".pickle")
        with open(path_model, "rb") as pickle_reader:
            model = pickle.load(pickle_reader)
        # Compute predictions utilizing the fitted machine learning model
        preds = model.predict_proba(features)
        # Parse predictions and add to cache
        df = pd.DataFrame(data={c: preds[:, 1]})
        preds_list.append(df)
    # Concat cached predictions together
    preds_final = pd.concat([dt_pred["ID"]] + preds_list, axis=1, sort=False)

    # Clean up results and create submission result
    preds_final = preds_final[["ID"] + cols]
    preds_final.ID = preds_final.ID.astype(float)
    preds_final = preds_final.sort_values("ID")

    # Export to disk
    path_results = os.path.join(path_preds, "ensemble.inference." + mode + \
                                "." + ml + ".csv")
    preds_final.to_csv(path_results, index=False)

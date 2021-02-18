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
# Provide pathes to prediction data
path_preds = "preds"

#-----------------------------------------------------#
#              AUCMEDI Ensemble for RIADD             #
#-----------------------------------------------------#
cols = ["DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM", "LS", "MS",
        "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST", "AION", "PT",
        "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "OTHER"]

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


# Iterate over all ensemble learning preidctions
dt_pred = None
for pred_file in sorted(os.listdir(path_preds)):
    if not pred_file.split(".")[2] == "labels" : continue
    if not pred_file.split(".")[3] == "simple" : continue
    # Load label prediction
    pred = pd.read_csv(os.path.join(path_preds, pred_file), sep=",", header=0)
    # Rename columns
    prefix = ".".join(pred_file.split(".")[0:2])
    label_cols = list(pred.columns[1:])
    label_cols = [prefix + "." + label for label in label_cols]
    pred.columns = ["ID"] + label_cols
    # Merge predictions
    if dt_pred is None : dt_pred = pred
    else : dt_pred = dt_pred.merge(pred, on="ID")

# Obtain features table
features = dt_pred.drop("ID", axis=1).to_numpy()

# Load random forest model
path_model = os.path.join("models", "ensemble", "labels.rf.pickle")
with open(path_model, "rb") as pickle_reader:
    model = pickle.load(pickle_reader)

# Run predictions using the RF model
preds = model.predict_proba(features)

# Extract probabilities
preds_list = []
for i, p in enumerate(preds):
    df = pd.DataFrame(data={cols[i]: p[:, 1]})
    preds_list.append(df)
preds_final = pd.concat([dt_pred["ID"]] + preds_list, axis=1, sort=False)

# Create submission result
preds_final["Disease_Risk"] = 0
preds_final = preds_final[["ID", "Disease_Risk"] + cols]
preds_final.ID = preds_final.ID.astype(float)
preds_final = preds_final.sort_values("ID")

# Export
preds_final.to_csv("MISIT_results.csv", index=False)

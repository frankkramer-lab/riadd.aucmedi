#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2020 IT-Infrastructure for Translational Medical Research,    #
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
import numpy as np
from plotnine import *
# Sklearn libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
# AUCMEDI libraries
from aucmedi import input_interface

#-----------------------------------------------------#
#                    Configurations                   #
#-----------------------------------------------------#
path_models = "/home/mudomini/projects/riadd.aucmedi/archiv/run_rank30/models"
path_preds = "/home/mudomini/projects/riadd.aucmedi/archiv/run_rank30/preds"

path_riadd = "/home/mudomini/data/RIADD/Upsampled_Set/"
path_images = os.path.join(path_riadd, "images")
path_csv = os.path.join(path_riadd, "data.csv")

# Define label columns
cols = ["Disease_Risk", "DR", "ARMD", "MH", "DN", "MYA", "BRVO", "TSLN", "ERM",
        "LS", "MS", "CSR", "ODC", "CRVO", "TV", "AH", "ODP", "ODE", "ST",
        "AION", "PT", "RT", "RS", "CRS", "EDN", "RPEC", "MHL", "RP", "OTHER"]

#-----------------------------------------------------#
#                 Plot fitting curve                  #
#-----------------------------------------------------#
# Collect fitting logs
dt_logs = None
# Iterate over all classifier architectures
for model_subdir in os.listdir(path_models):
    # Skip all non classifier model subdirs
    if not model_subdir.startswith("classifier_") : continue
    # Identify architecture
    arch = model_subdir.split("_")[1]
    path_arch = os.path.join(path_models, model_subdir)
    # Iterate over each fold of the CV
    for i in range(0, 5):
        path_fitting_log = os.path.join(path_arch, "cv_" + str(i) + ".logs.csv")
        dt_log_cv = pd.read_csv(path_fitting_log, sep=",", header=0)
        dt_log_cv.columns = ["epoch", "auc", "accuracy", "loss",
                             "val_auc", "val_accuracy", "val_loss"]
        dt_log_cv["architecture"] = arch
        dt_log_cv["fold"] = str(i)
        dt_log_cv["model"] = "classifier"
        if dt_logs is None : dt_logs = dt_log_cv
        else : dt_logs = pd.concat([dt_logs, dt_log_cv], axis=0)
# Iterate over all detector architectures
for model_subdir in os.listdir(path_models):
    # Skip all non detector model subdirs
    if not model_subdir.startswith("detector_") : continue
    # Identify architecture
    arch = model_subdir.split("_")[1]
    path_arch = os.path.join(path_models, model_subdir)
    # Iterate over each fold of the CV
    for i in range(0, 5):
        path_fitting_log = os.path.join(path_arch, "cv_" + str(i) + ".logs.csv")
        dt_log_cv = pd.read_csv(path_fitting_log, sep=",", header=0)
        dt_log_cv.columns = ["epoch", "auc", "accuracy", "loss",
                             "val_auc", "val_accuracy", "val_loss"]
        dt_log_cv["architecture"] = arch
        dt_log_cv["fold"] = str(i)
        dt_log_cv["model"] = "detector"
        if dt_logs is None : dt_logs = dt_log_cv
        else : dt_logs = pd.concat([dt_logs, dt_log_cv], axis=0)
# Melt fitting logs
dt_logs = dt_logs.melt(id_vars=["model", "architecture", "epoch", "fold"],
                       value_vars=["loss", "val_loss"],
                       var_name="dataset", value_name="score")
# Rename some values
dt_logs.replace({"classifier": "Disease Label Classifier",
                 "detector": "Disease Risk Detector"}, inplace=True)

# Plot fitting curves
fig = (ggplot(dt_logs, aes("epoch", "score", color="architecture",
                           linetype="factor(dataset)"))
           + geom_smooth(method="loess", size=2)
           + ggtitle("Fitting course during Training")
           + facet_wrap("model", scales="free")
           + xlab("Epoch")
           + ylab("Focal Loss")
           + scale_colour_discrete(name="Architecture")
           + scale_linetype_discrete(name="Dataset", labels=["Training",
                                                             "Validation"])
           + theme_bw(base_size=36))
# Store figure to disk
fig.save(filename="plot.fitting_course.png",
         path="./", width=20, height=10, dpi=200, limitsize=False)

#-----------------------------------------------------#
#            Prepare Performance Analysis             #
#-----------------------------------------------------#
# Iterate over all ensemble learning predictions
dt_pred = None
for pred_file in sorted(os.listdir(path_preds)):
    if not pred_file.split(".")[3] == "ensemble_train" : continue
    # Load label prediction
    pred = pd.read_csv(os.path.join(path_preds, pred_file), sep=",", header=0)
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
dt_pred = data.drop(["gt_" + s for s in cols], axis=1)

# Compute ROC & AUROC for each class (classifier & detector)
list_auroc = []
list_roc = []
for c in cols:
    c_gt = data["gt_" + c]
    for k in dt_pred.columns:
        if not k.endswith(c) : continue
        ks = k.split(".")
        c_pd = dt_pred[k]
        # Compute AUROC
        auc = roc_auc_score(c_gt, c_pd)
        list_auroc.append([ks[0], ks[1], ks[2], ks[3], auc])
        # Compute ROC
        fpr, tpr, _ = roc_curve(c_gt, c_pd)
        roc = pd.DataFrame({"model": ks[0], "architecture": ks[1], "fold": ks[2],
                            "class": ks[3], "fpr": fpr, "tpr": tpr})
        list_roc.append(roc)

#-----------------------------------------------------#
#    Compute inferences for ensembler via 5-fold CV   #
#-----------------------------------------------------#
# Load sampling from disk
with open(os.path.join(path_models, "sampling.json"), "r") as json_reader:
    sampling_dict = json.load(json_reader)
subsets = []
k_fold = 5
for i in range(0, k_fold):
    fold = "cv_" + str(i)
    x_train = np.array(sampling_dict[fold]["x_train"])
    y_train = np.array(sampling_dict[fold]["y_train"])
    x_val = np.array(sampling_dict[fold]["x_val"])
    y_val = np.array(sampling_dict[fold]["y_val"])
    subsets.append((x_train, y_train, x_val, y_val))

# Iterate over each fold in the k-fold Cross-Validation
for k in range(0, k_fold):
    # Load sampling
    (index_train, _, index_val, _) = subsets[k]
    train_data = data[data.ID.isin(index_train)]
    val_data = data[data.ID.isin(index_val)]
    # Preprocess data
    train_x = train_data.drop(["gt_" + s for s in cols] + ["ID"], axis=1).to_numpy()
    train_y = train_data[["gt_" + s for s in cols]].to_numpy()
    val_x = val_data.drop(["gt_" + s for s in cols] + ["ID"], axis=1).to_numpy()
    val_y = val_data[["gt_" + s for s in cols]].to_numpy()

    # Iterate over each class
    for i, c in enumerate(cols):
        print("Apply ensembling model:", "fold_" + str(k), c)
        c_gt = val_y[:,i]
        # Run Logistic Regression
        model = LogisticRegression(class_weight="balanced", max_iter=1000)
        model.fit(train_x, train_y[:,i])
        lr_preds = model.predict_proba(val_x)[:, 1]
        # Evaluate Performance for Logistic Regression
        auc = roc_auc_score(c_gt, lr_preds)
        list_auroc.append(["ensemble", "LogisticRegression", "cv_" + str(k), c, auc])
        fpr, tpr, _ = roc_curve(c_gt, lr_preds)
        roc = pd.DataFrame({"model": "ensemble", "architecture": "LogisticRegression",
                            "fold": "cv_" + str(k), "class": c, "fpr": fpr, "tpr": tpr})
        list_roc.append(roc)
        # Run Random Forest model
        model = RandomForestClassifier(class_weight="balanced", n_estimators=150)
        model.fit(train_x, train_y[:,i])
        rf_preds = model.predict_proba(val_x)[:, 1]
        # Evaluate Performance for Random Forest
        auc = roc_auc_score(c_gt, rf_preds)
        list_auroc.append(["ensemble", "RandomForest", "cv_" + str(k), c, auc])
        fpr, tpr, _ = roc_curve(c_gt, rf_preds)
        roc = pd.DataFrame({"model": "ensemble", "architecture": "RandomForest",
                            "fold": "cv_" + str(k), "class": c, "fpr": fpr, "tpr": tpr})
        list_roc.append(roc)

#-----------------------------------------------------#
#                Performance Evaluation               #
#-----------------------------------------------------#
# Collect evaluation data
dt_auroc = pd.DataFrame(list_auroc, columns=["model", "architecture", "fold",
                                             "class", "auroc"])
dt_roc = pd.concat(list_roc, axis=0, sort=False)

# Define a labeller function
def label_function_individual(label):
    label = label.replace("classifier", "Classifier : ")
    label = label.replace("detector", "Detector : ")
    label = label.replace("ensemble", "Ensembler : ")
    label = label.replace("EfficientNetB4", "EfficientNetB4 : ")
    label = label.replace("DenseNet201", "DenseNet201 : ")
    label = label.replace("ResNet152", "ResNet152 : ")
    label = label.replace("InceptionV3", "InceptionV3 : ")
    label = label.replace("RandomForest", "Random Forest : ")
    label = label.replace("LogisticRegression", "Logistic Regression : ")
    label = label.replace("cv_0", "Fold-0")
    label = label.replace("cv_1", "Fold-1")
    label = label.replace("cv_2", "Fold-2")
    label = label.replace("cv_3", "Fold-3")
    label = label.replace("cv_4", "Fold-4")
    return label

# Plot ROC results individually
fig = (ggplot(dt_roc, aes("fpr", "tpr", color="class"))
           + geom_line(size=1.5)
           + geom_abline(intercept=0, slope=1, color="black",
                         linetype="dashed")
           + ggtitle("Model Performance Overview")
           + facet_wrap("model + architecture + fold", labeller=label_function_individual)
           + xlab("False Positive Rate")
           + ylab("True Positive Rate")
           + scale_x_continuous(limits=[0, 1])
           + scale_y_continuous(limits=[0, 1])
           + scale_color_discrete(name="Classification")
           + theme_bw(base_size=28))
# Store figure to disk
fig.save(filename="plot.ROC.png", path="./", width=60, height=20, dpi=200,
         limitsize=False)

# Define a labeller function
def label_function_smoothed(label):
    label = label.replace("classifier", "Classifier : ")
    label = label.replace("detector", "Detector : ")
    label = label.replace("ensemble", "Ensembler : ")
    label = label.replace("EfficientNetB4", "EfficientNetB4")
    label = label.replace("DenseNet201", "DenseNet201")
    label = label.replace("ResNet152", "ResNet152")
    label = label.replace("InceptionV3", "InceptionV3")
    label = label.replace("RandomForest", "Random Forest")
    label = label.replace("LogisticRegression", "Logistic Regression")
    return label

fig = (ggplot(dt_roc, aes("fpr", "tpr", color="class"))
           + geom_smooth(method="gpr", size=1.5)
           + geom_abline(intercept=0, slope=1, color="black",
                         linetype="dashed")
           + ggtitle("Model Performance Overview")
           + facet_wrap("model + architecture", labeller=label_function_smoothed, ncol=4, nrow=2)
           + xlab("False Positive Rate")
           + ylab("True Positive Rate")
           + scale_x_continuous(limits=[0, 1])
           + scale_y_continuous(limits=[0, 1])
           + scale_color_discrete(name="Classification")
           + theme_bw(base_size=28))
# Store figure to disk
fig.save(filename="plot.ROC.cv_smoothed.png", path="./", width=50, height=20, dpi=200,
         limitsize=False)

# Store complete evaluation data to disk
dt_auroc.to_csv("eval_data.complete.csv", sep=",", index=False)

# Compute cross-validation averages
dt_auroc_avg = dt_auroc.groupby(["model", "architecture", "class"]).mean()
dt_auroc_avg.to_csv("eval_data.cv_avg.csv", sep=",", index=True)

# Compute class averages
dt_auroc_avg = dt_auroc.groupby(["model", "architecture"]).mean()
dt_auroc_avg.to_csv("eval_data.cv_class_avg.csv", sep=",", index=True)

#-----------------------------------------------------#
#                  Label Distribution                 #
#-----------------------------------------------------#
# Obtain ground truth data
data_upsampled = data[["ID"] + ["gt_" + s for s in cols]]
# Remove gt_prefix
data_upsampled.columns = ["ID"] + cols
# Identify original dataset indicies
org_ds = []
index_list = data_upsampled["ID"].to_list()
for index in index_list:
    if index.isnumeric() : org_ds.append(index)
# Get original dataset
data_original = data_upsampled[data_upsampled["ID"].isin(org_ds)]
# Compute label frequency for original dataset
freq_org = data_original.iloc[:, 1:].sum(axis=0)
freq_org["Normal"] = sum(data_original["Disease_Risk"]==0)
# Compute label frequency for upsampled dataset
freq_ups = data_upsampled.iloc[:, 1:].sum(axis=0)
freq_ups["Normal"] = sum(data_upsampled["Disease_Risk"]==0)
# Store label frequencies to disk
freq_org.to_csv("dataset.label_freq.original.csv", sep=",", index=True)
freq_ups.to_csv("dataset.label_freq.upsampled.csv", sep=",", index=True)

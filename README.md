# Multi-Disease Detection in Retinal Imaging based on Ensembling Heterogeneous Deep Learning Models

Participation at the Retinal Image Analysis for multi-Disease Detection Challenge

Copy & Paste Abstract

## Reproducibility

**Requirements:**
- Ubuntu 18.04
- Python 3.6
- NVIDIA TITAN RTX or a GPU with equivalent performance

**Step-by-Step workflow:**  

Adjust RIADD image directory path of all scripts in the 'configuration' section.

```sh
# Train detector models
python scripts/detector_DenseNet201.py
python scripts/detector_EfficientNetB4.py

# Train classifier models
python scripts/classifier_DenseNet201.py
python scripts/classifier_InceptionV3.py
python scripts/classifier_ResNet152.py
python scripts/classifier_EfficientNetB4.py

# Train Logistic Regression models
python scripts/ensemble_training.py

# Run Inference
python scripts/inference.py
python scripts/ensemble.py

# Perform result evaluation
python scripts/evaluation.py
```

## Dataset: RIADD

**Reference:** https://riadd.grand-challenge.org/Home/

The Retinal Image Analysis for Multi-Disease Classification (RIADD) is a challenge from the ISBI 2021. The aim is to multi-label classify different sized retinal microscrope images.

Microscope distribution:  
{(1424, 2144, 3): 1493, (1536, 2048, 3): 150, (2848, 4288, 3): 277}

![fig_LabelFreq](docs/label_freq.png)

This dataset consists of diseases/abnormalities (diabetic retinopathy (DR), age-related macular degeneration (ARMD), media haze (MZ), drusen (DN), myopia (MYA), branch retinal vein occlusion (BRVO), tessellation (TSLN), epiretinal membrane (ERM), laser scar (LS), macular scar (MS), central serous retinopathy (CSR), optic disc cupping (ODC), central retinal vein occlusion (CRVO), tortuous vessels (TV), asteroid hyalosis (AH), optic disc pallor (ODP), optic disc edema (ODE), shunt (ST), anterior ischemic optic neuropathy (AION), parafoveal telangiectasia (PT), retinal traction (RT), retinitis (RS), chorioretinitis (CRS), exudation (EDN), retinal pigment epithelium changes (RPEC), macular hole (MHL), retinitis pigmentosa (RP), cotton wool spots (CWS), coloboma (CB), optic disc pit maculopathy (ODPM), preretinal hemorrhage (PRH), myelinated nerve fibers (MNF), hemorrhagic retinopathy (HR), central retinal artery occlusion (CRAO), tilted disc (TD), cystoid macular edema (CME), post traumatic choroidal rupture (PTCR), choroidal folds (CF), vitreous hemorrhage (VH), macroaneurysm (MCA), vasculitis (VS), branch retinal artery occlusion (BRAO), plaque (PLQ), hemorrhagic pigment epithelial detachment (HPED) and collateral (CL)) based on their visual characteristics as shown in the Figure below.

![fig_classes](docs/All_disease_image_ahICpUG.png)

## Methods

The implemented medical image classification pipeline can be summarized in the following core steps:
- Class Weighted Focal Loss and Upsampling to conquer Class Imbalance
- Stratified Multi-label 5-fold Cross-Validation
- Extensive real-time image augmentation
- Multiple Deep Learning Model Training
- Distinct Training for Multi-Disease Labels and Disease Risk Detection
- Ensemble Learning Strategy: Bagging & Stacking
- Stacked Binary Logistic Regression Models for Distinct Classification

![fig_pipeline](docs/RIADD_aucmedi.png)

This pipeline was based on AUCMEDI, which is an in-house developed open-source framework to setup complete medical image classification pipelines with deep learning models on top of Tensorflow/Keras⁠. The framework supports extensive preprocessing, image augmentation, class imbalance strategies, state-of-the-art deep learning models and ensemble learning techniques. The experiment was performed in parallel with multiple NVIDIA TITAN RTX GPUs.

## Results & Discussion

todo

![fig_results](docs/plot.ROC.png)

## Author

Dominik Müller  
Email: dominik.mueller@informatik.uni-augsburg.de  
IT-Infrastructure for Translational Medical Research  
University Augsburg  
Bavaria, Germany

## How to cite / More information

Coming soon.

```
Coming soon.
```

Thank you for citing our work.

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE Version 3.  
See the LICENSE.md file for license rights and limitations.

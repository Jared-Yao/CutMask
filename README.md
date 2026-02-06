# CutMask

Enhanced Industrial Anomaly Detection via
CutMask Data Augmentation: A Self-Supervised
Approach
![alt text](<pic/Schematic diagram.png>)



## ♻️ Reproducibility & Citation

The archived, citable version of this code is available on **Zenodo**: **DOI: 10.5281/zenodo.18503408**.

If you use this code or the associated artifacts, please cite  **(1)** the Zenodo record (**10.5281/zenodo.18503408**).


# 1 Set Up
## 1.1 Experimental platform and environment

- **CPU**:14 vCPU Intel(R) Xeon(R) Gold 6330 CPU @ 2.00GHz
- **GPU**: NVIDIA GeForce RTX 3090  
- **System**: ubuntu20.04
- **Python**: 3.8  
- **torch**: 1.11.0 

Install the following requirements:
1. Pytorch and torchvision
2. sklearn
3. pandas
4. seaborn
5. tqdm
6. tensorboard
7. Pillow

For example with [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html):
```
conda create -n cutmask python=3.8
conda activate cutmask
```
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
```
pip install -r requirements.txt
```

## 1.2 Datasets
Download the MVTec Anomaly detection Dataset from [here](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract it into a new folder named `Data`.

Uncompressed directory structure:

```shell
Data
|-- bottle
|-----|----- ground_truth
|-----|----- test
|-----|--------|------ good
|-----|--------|------ broken_large
|-----|--------|------ ...
|-----|----- train
|-----|--------|------ good
|-- cable
|-- ...
```

containing in total 15 subdatasets: `bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`,
`leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`, `wood`, `zipper`.

Download the VisA Dataset from [here](https://github.com/amazon-science/spot-diff) and extract it into a new folder named `Data`.

## 1.3 training hyperparameters

- **Learning rate**: 0.001
- **Optimizer**: SGD  
- **Epochs**: 400  
- **Batch size**: 32 
- **Image size**: 256*256
- **test_epochs**:10
- **head_layer**:2

## 1.4 Document Description

- **The model configuration file is stored in**: `models`  
- **run_training.py**: Script for training the model
- **make_gradcam_heatmap.py**: Script for generating heat maps
- **eval.py**: Script for evaluating the model


# 2 Usage

## 2.1 Run Training
The Script will train a model for each defect type and save it in the `model_dir` Folder.
```
python run_training.py --model_dir models --head_layer 2
```


To enable training on an Nvidia GPU use the `--cuda 1` flag.
```
python run_training.py --model_dir models --head_layer 2 --cuda 1
```

One can track the training progress of the models with tensorboard:
```
tensorboard --logdir logdirs
```

Single-class training (example)
```
python train.py --type bottle --epochs 400 --cuda True
```
All-category training
```
python train.py --all --epochs 400 --cuda True
```



Different variant training commands:choices=['normal', 'scar', '3way', 'union', "mask3way", "masknormal", "maskscar"]
```
python train.py --type bottle --variant masknormal --epochs 400 --cuda True

```
Similarly, you can set epochs, test_epochs, learning rate, optimizer and other parameters in the configuration file.
```
python train.py --type bottle --variant masknormal --epochs 400 --test_epochs 10 --lr 0.01 --optim sgd --cuda True
```

## 2.2 Run Evaluation
This will create a new directory `Eval` with plots for each defect type/model.
```
python eval.py --model_dir models --head_layer 2
```


# 3 Description of the key algorithm


## 3.1 CutMask

### CutMaskNormal

**Code location:** `cutpaste1.py`

**Description:**
CutMaskNormal extends CutPasteNormal by introducing a prior mask bank from `maskpackge1/`. It still crops a rectangular patch by randomly sampling the area ratio and aspect ratio; however, before pasting, the patch is “carved out” using the mask’s alpha map. As a result, the pseudo-anomalous region exhibits a non-rectangular contour, which reduces the characteristic *rectangular block artifacts* of CutPaste and improves both the realism and diversity of synthesized anomalies.

---

### CutMaskScar

**Code location:** `cutpaste1.py`

**Description:**
CutMaskScar targets *scratch-like or elongated* anomaly patterns. It first crops a relatively slender patch and applies a prior mask to carve its shape, then performs a random rotation. If the rotated patch exceeds the image boundaries, an adaptive rescaling step is applied to ensure valid pasting. Compared with standard scar augmentation, CutMaskScar further constrains boundary geometry via the mask, making synthesized anomalies closer to real defect contours.

---

### CutMask3Way

**Code location:** `cutpaste1.py`

**Description:**
CutMask3Way is designed for **three-way** pretext classification. For each normal image, it simultaneously constructs three views: the original image, a CutMaskNormal-augmented view, and a CutMaskScar-augmented view. The model is trained to discriminate among these views, leading to stronger representations—particularly improved sensitivity to anomalies with different morphological characteristics.





## 3.2 Self-Supervised Patch-wise Channel Attention Block（SSPCAB）

**Code location:** `sspcab_torch.py`

**Description:**
SSPCAB is a lightweight feature enhancement module that combines **four-branch spatial sampling** with **channel attention**, aiming to improve feature discriminability without introducing substantial computational overhead. Its core idea is to apply padding to the feature map and then sample four spatially shifted sub-regions, each processed by a separate convolution to capture responses from different spatial positions. The four responses are summed and passed through a nonlinearity to form an aggregated spatially enhanced representation. Finally, an **SE (Squeeze-and-Excitation)** channel attention mechanism re-calibrates channel-wise importance, thereby emphasizing anomaly-relevant channels and suppressing redundant background channels.


## 3.3 CAFL

**Code location:** `cafl_loss.py`

**Description:**
CAFL is designed for self-supervised pretraining in anomaly detection (e.g., **2-way** or **3-way** classification under CutPaste/CutMask). It unifies three objectives into a single loss function to address common industrial anomaly challenges, including **small defect regions**, **scarce hard samples**, and **confusability between positive and negative samples**:

1. **CFL :**
   Leverages (asymmetric) focal reweighting to emphasize hard-to-classify samples, and optionally employs a **cyclical schedule** across training stages to improve robustness.

2. **Adversarial Loss :**
   Enhances separability between **normal** and **synthetic anomalies**, making the model more sensitive to pseudo-anomalous signals.

3. **Self-attention auxiliary loss :**
   Regularizes SSPCAB-enhanced features to encourage **beneficial yet non-distortive** attention refinement, preventing excessive feature drift.






## 3.4 ProjectionNet




**Code location:** `model1.py`

**Description:**
ProjectionNet is a feature extraction network tailored for self-supervised pretraining under CutPaste/CutMask. It adopts ResNet18 as the backbone to extract high-level semantic representations, inserts an SSPCAB attention-enhancement module on the final feature map, and then applies global pooling to obtain the embedding. A subsequent MLP projection head plus a classification layer produces logits for 2-way/3-way pretext classification. Meanwhile, the network computes an SSPCAB auxiliary regularization loss (the MSE between the enhanced feature and the original feature), which aligns with the **$\beta L_{\text{SSPCAB}}$** term in CAFL.




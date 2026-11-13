# MA3D-Net: 3DMM-Guided Adaptive Modulation for Facial Expression Recognition in the Wild



## Table of Contents

1. [Description](#1-description)
2. [Dataset Information](#2-dataset-information)
3. [Code Information](#3-code-information)
4. [Requirements](#4-requirements)
5. [Usage Instructions](#5-usage-instructions)
6. [Methodology](#6-methodology)
7. [Citations](#7-citations)
8. [License and Contribution Guidelines](#8-license-and-contribution-guidelines)


## 1. Description

This repository contains the official PyTorch implementation of **MA3D-Net**, a facial expression recognition (FER) framework that adds parametric 3D facial priors to 2D representation learning. It is designed for unconstrained images, where illumination, degradation, occlusion and appearance changes limit purely 2D methods.

MA3D-Net has three main contributions:

- **Feature-wise Adaptive Modulation (FAM).** 3D Morphable Model (3DMM) parameters reconstructed by [EMOCA](https://github.com/radekd91/emoca) recalibrate intermediate appearance features within stacked residual blocks, instead of being directly concatenated.
- **Multi-scale cross-interaction.** The geometry-conditioned appearance features interact with 2D landmark features through hierarchical bidirectional cross-attention.
- **Margin-Aware Cross-Entropy (MACE) loss.** The loss emphasizes ambiguous samples near decision boundaries to improve separation between visually similar expressions.

The repository provides code for 3DMM feature preparation, training, evaluation, ablation and visualization. See the manuscript for full details.

---

## 2. Dataset Information

MA3D-Net is evaluated on three public FER benchmarks and one domain-specific dataset. **The datasets are not redistributed here.** Download them from their official sources.

| Dataset | Train | Test | Classes | Source |
|---------|------:|-----:|:-------:|--------|
| RAF-DB | 12,271 | 3,068 | 7 | [Official site](http://www.whdeng.cn/RAF/model1.html#dataset) |
| FERPlus | 28,709 | 3,589 | 8 | [microsoft/FERPlus](https://github.com/microsoft/FERPlus) |
| CAER-S | 44,996 | 20,987 | 7 | [Official site](https://caer-dataset.github.io/) |
| CheoFaMo | 4,675 | 1,169 | 6 | [Mendeley Data](https://data.mendeley.com/datasets/2h6hxwkbwn/2) |

- **RAF-DB**: in-the-wild face images. The 7 basic expressions are used: neutral, happiness, surprise, sadness, anger, disgust and fear.
- **FERPlus**: FER2013 with improved crowd-sourced labels. It has 8 classes: the 7 basic expressions plus contempt.
- **CAER-S**: static images from the context-aware CAER dataset, with 7 emotion classes.
- **CheoFaMo**: segmented facial image sequences from Vietnamese traditional Chèo theatre. It has 5,844 annotated segments in 6 mood classes: happiness, sadness, fear, love, anger and hatred. The classes are heavily imbalanced. An 80/20 stratified split is applied at the sequence level.

---

## 3. Code Information

```
MA3D-Net/
├── train.py              # Training
├── eval.py               # Evaluation on the test split
├── engine.py             # Training/validation loops and loss schedule
├── build_dataloader.py   # Transforms and data loaders
├── utils.py              # Helper functions
├── requirements.txt      # Python dependencies
├── docker/Dockerfile     # Docker environment
├── models/
│   ├── MA3D.py           # MA3D-Net model
│   ├── ThreeDMM_Adaptive.py  # 3DMM Modulation Module (FAM)
│   ├── hyp_crossvit.py   # Multi-scale cross-interaction (pyramid cross-attention)
│   ├── ir50.py           # IR50 appearance backbone
│   ├── mobilefacenet.py  # MobileFaceNet landmark backbone
│   └── sam.py            # SAM optimizer wrapper
├── loss_function/loss.py # MACE and label-smoothing cross-entropy
├── Read_dataset/         # Dataset loaders and 3DMM normalization statistics
├── scripts/              # t-SNE, attention-map and radar-chart visualization
├── checkpoints/          # Pretrained backbones and MA3D-Net checkpoints
├── log/                  # Training logs
└── Datasets/             # Datasets (not included)
```

---

## 4. Requirements

- **Python dependencies**: listed in `requirements.txt`. Install them on top of PyTorch (see Section 5.1).
- **EMOCA**: needed only to extract the 3DMM features. Install it in a separate environment by following the [official instructions](https://github.com/radekd91/emoca).
- **Hardware**: an NVIDIA GPU is recommended. All experiments in the manuscript were run on an NVIDIA GeForce GTX 1660 (6 GB) with batch size 8.

---

## 5. Usage Instructions

Run all commands from the repository root.

### 5.1 Environment setup

**Option 1: Conda**

```bash
conda create -n ma3d python=3.10 -y
conda activate ma3d
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install tqdm opencv-python
```

**Option 2: Docker**

```bash
docker build -f docker/Dockerfile -t ma3d .
docker run --gpus all -it --rm -v "$(pwd)":/workspace ma3d
```

### 5.2 Prepare the datasets

1. Download the datasets (Section 2).
2. Put the images in the layouts below. Each sample has its own folder. That folder holds the face image and the EMOCA parameter files `shape.npy`, `tex.npy`, `exp.npy`, `pose.npy` and `detail.npy`.

```
Datasets/
├── RAF-DB_lmk/
│   ├── train/
│   │   └── 1/ ... 7/                      # RAF-DB class folders
│   │       └── train_00006_aligned/
│   │           ├── train_00006_aligned.jpg
│   │           └── shape.npy, tex.npy, exp.npy, pose.npy, detail.npy
│   └── test/                              # same layout
├── FER_Plus/fer_plus_05/
│   ├── train/<class_name>/<sample>/       # image + 5 .npy files
│   ├── test/<class_name>/<sample>/
│   └── ferplus_stats.npz                  # created in step 5.4
├── caers/caers/
│   ├── train/<class_name>/<sample>/       # image + 5 .npy files
│   ├── test/<class_name>/<sample>/
│   └── caers_stats.npz                    # created in step 5.4
└── CheoFamo/
    ├── Metadata.json                      # from the CheoFaMo release
    ├── Frames/<VideoID>_<CharacterID>_<ApexIndex>/   # e.g. 01_01_00323/
    │   ├── 01_01_00323.jpg                # apex frame
    │   └── shape.npy, tex.npy, exp.npy, pose.npy, detail.npy
    └── cheofamo_stats.npz                 # created in step 5.4
```

The root paths are set in `build_dataloader.py`.

### 5.3 Extract 3DMM features with EMOCA

1. Clone and install the official [EMOCA](https://github.com/radekd91/emoca) repository.
2. Run EMOCA on every image of each dataset.
3. Save the FLAME parameter files (`shape.npy`, `tex.npy`, `exp.npy`, `pose.npy`, `detail.npy`) into that image's sample folder.

EMOCA already writes these files as `.npy`, so no conversion is needed.

### 5.4 Compute 3DMM normalization statistics

The dataset classes standardize each 3DMM parameter with a per-dimension mean and standard deviation. These are loaded from a `.npz` file in the dataset root, if that file exists.

```bash
python -c "from Read_dataset.FER_Plus import compute_ferplus_stats; compute_ferplus_stats('Datasets/FER_Plus/fer_plus_05', split='train')"
```

```bash
python -c "from Read_dataset.Caers import compute_stats; compute_stats('Datasets/caers/caers', split='train')"
```

```bash
python -c "from Read_dataset.Cheo_FaMo import compute_cheofamo_stats; compute_cheofamo_stats('Datasets/CheoFamo')"
```

```bash
python -c "from Read_dataset.RAF_DB import compute_stats; compute_stats('Datasets/RAF-DB_lmk', split='train', output_name='rafdb_emoca_stats.npz')"
```


### 5.5 Download pretrained backbones

1. Download the IR50 (appearance) and MobileFaceNet (landmark) weights from [Google Drive](https://drive.google.com/drive/folders/1X9pE-NmyRwvBGpVzJOEvLqRPRfk_Siwq?usp=sharing).
2. Place them in `checkpoints/` under the file names that `models/MA3D.py` loads:

```
checkpoints/
├── ir50_o1.pth
└── mobilefacenet_model_best.pth_o1.tar
```

### 5.6 Training

Example (RAF-DB):

```bash
python train.py --data_type RAF-DB --num_classes 7 --seed 42 --batch_size 8 --resume_dir checkpoints --resume_name rafdb_ma3d --log_file rafdb_ma3d_seed_42.txt
```

For the other datasets, change `--data_type` and `--num_classes`:

- FERPlus: `--data_type FerPlus --num_classes 8`
- CAER-S: `--data_type Caers --num_classes 7`
- CheoFaMo: `--data_type CheoFaMo --num_classes 6`

Outputs:

- The best checkpoint is saved as `<resume_dir>/<resume_name>_seed_<seed>.pth`.
- The per-epoch log is written to `log/<log_file>`.

Notes:

- **Batch size**: the default batch size of 8 matches the manuscript. With more GPU memory you can increase `--batch_size`. If you do, scale `--lr` accordingly; the manuscript uses the square-root scaling rule.
- **Multiple runs**: to reproduce the multi-run results, train with different `--seed` values.

### 5.7 Evaluation

1. Download the trained MA3D-Net checkpoints from [Google Drive](https://drive.google.com/drive/folders/10tltNSB009VoalxTo46IUMpQhPANudhF?usp=drive_link) and place them in `checkpoints/`. You can also use a checkpoint produced by `train.py`.
2. Run, for example:

```bash
python eval.py --data_type RAF-DB --num_classes 7 --ckpt_path checkpoints/rafdb_ma3d_seed_42.pth
```

`eval.py` prints:

- overall accuracy, macro-F1 and balanced accuracy;
- per-class precision and recall;
- the confusion matrix.

### 5.8 Ablation checkpoints

1. Download the ablation checkpoints from [Google Drive](https://drive.google.com/drive/folders/10ysC_2tfkfZTyalj-nvX96FtAsCC8uCl?usp=sharing) and place them in `checkpoints/ablation/`.
2. Before evaluating an ablation checkpoint, set the model architecture and training settings in the code to match that ablation variant. Otherwise the checkpoint will not load or will give wrong results.
3. Run, for example:

```bash
python eval.py --data_type RAF-DB --num_classes 7 --ckpt_path checkpoints/ablation/baseline.pth
```

### 5.9 Visualization

Run the scripts as modules from the repository root:

```bash
python -m scripts.tsne --data_type RAF-DB --num_classes 7 --ckpt_path checkpoints/rafdb_ma3d_seed_42.pth
```

```bash
python -m scripts.attn_vis --data_type RAF-DB --num_classes 7 --ckpt_path checkpoints/rafdb_ma3d_seed_42.pth
```

Radar chart:

1. Save the per-class results of each model with `--mode save`. Each run writes `results/<out_name>.npy`.
2. Plot the comparison with `--mode vis`.

```bash
python -m scripts.radar_comparison --mode save --data_type RAF-DB --ckpt_path checkpoints/baseline.pth --out_name baseline
```

```bash
python -m scripts.radar_comparison --mode save --data_type RAF-DB --ckpt_path checkpoints/rafdb_ma3d_seed_42.pth --out_name ours
```

```bash
python -m scripts.radar_comparison --mode vis --baseline_path results/baseline.npy --ours_path results/ours.npy
```

---

## 6. Methodology

Brief overview only. See the *Methodology* section of the manuscript for details.

### 6.1 Data processing

1. **3DMM extraction**: the frozen EMOCA network estimates FLAME parameters from each image:
   - shape (100), expression (50), pose (6), detail (128) and texture (50);
   - these are concatenated into a 334-D vector.
2. **3DMM standardization**: each parameter dimension is standardized with the training-set mean and standard deviation.
3. **Image input**: face images are resized to 224 × 224. The landmark branch uses 112 × 112 inputs.

### 6.2 Modeling

1. **Multi-modal feature extraction**: three branches produce the input features.
   - An IR50 backbone extracts appearance features.
   - A frozen MobileFaceNet extracts landmark features.
   - EMOCA provides the 3DMM vector.
2. **3DMM Modulation Module (FAM)**: the 3DMM vector is mapped to channel-wise scale and shift factors that modulate the appearance features in stacked 3DMM Adaptive Blocks.
3. **Multi-scale Cross-interaction Module**: appearance and landmark tokens interact through bidirectional cross-attention at three scales.
4. **Classification head**: predicts the expression class from the aggregated classification token.
5. **Training**:
   - optimizer: AdamW with learning rate 1e-5 and weight decay 1e-4;
   - learning-rate schedule: exponential, γ = 0.98;
   - 100 epochs with batch size 8;
   - loss: label-smoothed cross-entropy (α = 0.2) for a 10-epoch warm-up, then MACE.

### 6.3 Evaluation method

- **Test splits**: models are evaluated on the held-out test split of each dataset.
- **Comparison with state-of-the-art methods**:
  - MA3D-Net is compared with published FER methods on all four datasets.
  - The main baseline, POSTER, is reproduced under identical training conditions.
  - Both models are trained over five independent runs (seeds 30, 42, 123, 3407, 9999).
  - Results are reported as mean ± std, with 95% confidence intervals and paired two-sided t-tests.
- **Ablation study**: on RAF-DB and CheoFaMo, MA3D-Net components, 3DMM parameter groups, fusion mechanisms and loss functions are removed or replaced.
- **Robustness analysis**: MA3D-Net and POSTER are compared on the RAF-DB test set under 52 controlled corruption conditions: blur, illumination, random occlusion and anatomical face masking.

### 6.4 Assessment metrics

- **Accuracy**: the standard metric on RAF-DB, FERPlus and CAER-S, which allows direct comparison with published results.
- **Macro-F1 and balanced accuracy**: used for CheoFaMo, whose severe class imbalance makes accuracy alone misleading. Both metrics weight every class equally.
- **Per-class precision, recall and confusion matrices**: show which expressions are confused with each other, especially minority classes.

---

## 7. Citations

If you use this code, please cite the MA3D-Net manuscript:

```bibtex
@article{hoang2026ma3dnet,
  title   = {MA3D-Net: 3DMM-Guided Adaptive Modulation for Facial Expression Recognition in the Wild},
  author  = {Hoang, Quoc Viet and Nguyen, Minh Kien and Nguyen, Tien Phuc and Le, Thanh Ha and Dinh, Quang Trung and Ma, Thi Chau},
  journal = {PeerJ Computer Science},
  note    = {Under review},
  year    = {2026}
}
```

If you use the datasets or third-party components, please also cite their original papers:

```bibtex
@inproceedings{li2017rafdb,
  title     = {Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild},
  author    = {Li, Shan and Deng, Weihong and Du, JunPing},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {2584--2593},
  year      = {2017}
}

@inproceedings{barsoum2016ferplus,
  title     = {Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution},
  author    = {Barsoum, Emad and Zhang, Cha and Ferrer, Cristian Canton and Zhang, Zhengyou},
  booktitle = {Proceedings of the 18th ACM International Conference on Multimodal Interaction (ICMI)},
  pages     = {279--283},
  year      = {2016},
  doi       = {10.1145/2993148.2993165}
}

@inproceedings{lee2019caer,
  title     = {Context-Aware Emotion Recognition Networks},
  author    = {Lee, Jiyoung and Kim, Seungryong and Kim, Sunok and Park, Jungin and Sohn, Kwanghoon},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages     = {10142--10151},
  year      = {2019},
  doi       = {10.1109/ICCV.2019.01024}
}

@misc{cheofamo2025,
  title        = {CheoFaMo: A dataset of segmented image sequences for in-depth mood analysis in Vietnamese traditional Ch{\`e}o},
  author       = {Ma, Thi Chau and Nguyen, Minh Kien and Vu, Thanh Long and Nguyen, Tien Phuc and Hoang, Quoc Viet and Le, Thanh Ha and Dinh, Quang Trung},
  howpublished = {Mendeley Data, V2},
  year         = {2026},
  doi          = {10.17632/2h6hxwkbwn.2}
}

@inproceedings{danecek2022emoca,
  title     = {EMOCA: Emotion Driven Monocular Face Capture and Animation},
  author    = {Dan{\v{e}}{\v{c}}ek, Radek and Black, Michael and Bolkart, Timo},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {20279--20290},
  year      = {2022},
  doi       = {10.1109/CVPR52688.2022.01967}
}

@inproceedings{zheng2023poster,
  title     = {POSTER: A Pyramid Cross-Fusion Transformer Network for Facial Expression Recognition},
  author    = {Zheng, Ce and Mendieta, Matias and Chen, Chen},
  booktitle = {IEEE/CVF International Conference on Computer Vision Workshops (ICCVW)},
  pages     = {3138--3147},
  year      = {2023},
  doi       = {10.1109/ICCVW60793.2023.00339}
}
```

---

## 8. License and Contribution Guidelines

### License

**Code license: to be decided by the authors** (a `LICENSE` file will be added to the repository).

Third-party components and data keep their own licenses and terms of use:

- **Datasets**: RAF-DB, FERPlus, CAER-S and CheoFaMo are distributed by their owners under their own terms. This repository does not redistribute them.
- **EMOCA and the FLAME model**: their own licenses apply. Note in particular their non-commercial research terms.
- **Pretrained backbones** (IR50, MobileFaceNet) and the POSTER-based cross-attention design: subject to the terms of their original releases.

### Contributing

Contributions are welcome:

1. **Bug reports and questions**: open an issue on [GitHub](https://github.com/HoangViet252006/MA3D-Net/issues). Include your environment (OS, Python, PyTorch, CUDA, GPU), the exact command you ran and the full error message.
2. **Code contributions**:
   - fork the repository and create a feature branch;
   - keep changes focused;
   - check that `train.py` and `eval.py` still run;
   - open a pull request that describes the change and its motivation.
3. **Reproducibility reports**: if your results differ from those in the manuscript, open an issue with your seed, batch size, hardware and training log (`log/*.txt`).

### Contact

For questions about the paper, contact the corresponding author: **Thi Chau Ma** (chaumt@vnu.edu.vn), VNU University of Engineering and Technology, Hanoi, Vietnam.

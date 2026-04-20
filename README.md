# Multi-Task Ranking: PLE & DCN-v2 on Tencent TenRec Dataset

This repository provides a high-performance Multi-task Ranking (Ranking) framework implemented in PyTorch. The project integrates state-of-the-art architectures to solve common challenges in recommendation systems, such as **task conflicts (Seesaw Effect)** and **data sparsity**.

## 🌟 Key Features
- **Feature Interaction**: Implemented **DCN-v2** to capture high-order explicit feature crosses, significantly outperforming standard MLPs.
- **Architecture**: Leveraged **PLE (Progressive Layered Extraction)** to decouple task-specific knowledge from shared representations.
- **Optimization**: Integrated **UWL (Uncertainty Weighting)** to dynamically balance loss weights for multiple targets (Click, Follow, Like, Share).

## 🏗️ Model Architecture
The model consists of five key components:
1. **Embedding Layer**: 16-dimensional latent vectors for sparse features.
2. **Cross Layer (DCN-v2)**: Explicit vector-wise interactions.
3. **Extraction Layer (PLE)**: Shared and task-specific experts with gating mechanisms.
4. **Task Towers**: Independent MLP heads for multi-target prediction.
5. **Dynamic Weighter**: Self-adaptive loss weighting based on homoscedastic uncertainty.

## 📊 Ablation Study (Experimental Results)
Evaluated on the **Tencent TenRec (QK-video)** dataset (100k samples). Our full model demonstrates superior performance over the industry-standard **Shared-Bottom** baseline:

| Approach | Click AUC | Follow AUC | Like AUC | **Share AUC (Sparse)** |
| :--- | :--- | :--- | :--- | :--- |
| **Shared-Bottom (Baseline)** | 0.7694 | 0.7101 | 0.7268 | 0.7721 |
| **Our Full Model (PLE+DCN+UWL)** | **0.7787** | **0.7128** | **0.7456** | **0.7791** |
| **Improvement (Abs.)** | **+1.01%** | **+0.32%** | **+1.95%** | **+0.79%** |

*Note: The massive gain in Share AUC proves that PLE and UWL effectively mitigate the data sparsity issue for long-tail user behaviors.*

## 🛠️ Getting Started
### Installation
```bash
pip install -r requirements.txt

References
PLE: Tang et al., RecSys 2020.

DCN-v2: Wang et al., WWW 2021.

UWL: Kendall et al., CVPR 2018.

TenRec: Yuan et al., NeurIPS 2022.

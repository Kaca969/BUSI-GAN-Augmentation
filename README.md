BUSI Ultrasound Classification with GAN-Based Data Augmentation
Overview

This project investigates whether GAN-generated synthetic images can improve breast ultrasound lesion classification using the BUSI dataset. The workflow is divided into two stages: first, training a conditional GAN to generate realistic ultrasound images, and second, running controlled classification experiments to measure the impact of synthetic data augmentation. The objective is not to maximize absolute accuracy, but to isolate and quantify the effect of GAN-based augmentation in a controlled and reproducible setup.

Dataset

The BUSI dataset contains three classes: benign, malignant, and normal. All three classes are used during GAN training. For classification experiments, the task is reduced to a binary problem (benign vs. malignant). Mask files are ignored. The dataset is split at the image level rather than the case (patient) level. If multiple images belong to the same patient, this may introduce data leakage. A case-level split is identified as a recommended methodological improvement.

Stage 1 – Conditional GAN Training

The conditional GAN generates 128×128 grayscale ultrasound images conditioned on class labels. The generator starts from a 128-dimensional latent vector, projects it into an 8×8 feature map, and progressively upsamples using transposed convolutions, ending with a tanh activation (images normalized to [-1, 1]).

The discriminator follows a convolutional architecture where class embeddings are spatially expanded and concatenated with the input image. It outputs a single logit trained using BCEWithLogitsLoss.

To stabilize training, several standard techniques are applied: label smoothing for real samples (0.9), light Gaussian noise added to real images, and Adam optimizer with betas (0.5, 0.999). Synthetic image previews and generator checkpoints are saved after each epoch.

Stage 2 – Classification Experiments

A lightweight CNN classifier (four convolutional blocks with max pooling and dropout at 0.3) is trained for the benign vs. malignant task. The dataset is split 70/15/15 (train/validation/test), stratified by class labels. Two experiments are conducted:

REAL_ONLY — trained on real images only

REAL_PLUS_SYNTH — trained on real images combined with GAN-generated synthetic samples

Both models are evaluated on the same held-out test set using accuracy, confusion matrix, and a full classification report (precision, recall, F1-score).

Results

Accuracy (real only): 0.7755
Accuracy (real + synthetic): 0.7959
Improvement: +2.04 percentage points

The improvement is modest but consistent. Importantly, the baseline model is intentionally simple and does not use transfer learning. The rationale is that if synthetic augmentation provides measurable gains even in this minimal setup, stronger architectures would likely benefit as well.

Reproducibility

The experiments use a fixed random seed, deterministic dataset splits, a separate validation set for model selection, and a strictly untouched test set evaluated only at the final stage. The experimental setup prioritizes methodological clarity and clean evaluation practices.

Limitations

The image-level split is the primary methodological limitation. Additional limitations include the absence of FID or other quantitative metrics for synthetic image quality, lack of comparison with classical augmentation techniques, and the intentionally simple CNN architecture. Multi-seed evaluation (mean ± standard deviation) would further strengthen the robustness of the results.

Possible Improvements

Future extensions could include case-level data splitting, integration of a stronger backbone such as ResNet18 with transfer learning, direct comparison with classical augmentation pipelines, quantitative GAN evaluation using FID or similar metrics, and multi-seed experiments with variance reporting.

Project Purpose

The project emphasizes experimental clarity over raw performance. It demonstrates an end-to-end deep learning pipeline for medical image synthesis and classification, combining GAN-based augmentation with controlled evaluation. The structure is intentionally designed to be transparent, reproducible, and easy to extend for further research.

BUSI Ultrasound Classification with GAN-Based Data Augmentation
Overview
This project looks at whether GAN-generated synthetic images can actually help when classifying breast ultrasound lesions — using the BUSI dataset as the testbed. The workflow splits into two stages: first training a conditional GAN to produce realistic ultrasound images, then running controlled classification experiments to see if adding those synthetic images makes any measurable difference.
The point wasn't to chase the highest possible accuracy, but to isolate and measure the effect of synthetic augmentation in a controlled, reproducible setup.
Dataset
The BUSI dataset covers three classes — benign, malignant, and normal — all three of which are used during GAN training. For the classification experiments, only benign vs. malignant is considered (binary task). Mask files are ignored throughout.
One thing worth noting: the data split is done at the image level, not the case level. If multiple images come from the same patient, this could introduce some leakage — a case-level split would be a cleaner approach and is flagged as a recommended improvement.
Stage 1 – Conditional GAN Training
The GAN generates 128×128 grayscale ultrasound images conditioned on class labels.
The generator starts from a 128-dimensional latent vector, projects it to an 8×8 feature map, and upsamples through a series of transposed convolutions, ending with a tanh activation (images normalized to [-1, 1]). The discriminator uses a convolutional architecture where class embeddings are spatially expanded and concatenated with the input image, outputting a single logit via BCEWithLogitsLoss.
A few stabilization tricks are in place: label smoothing for real samples (0.9), light Gaussian noise added to real images during training, and Adam with betas (0.5, 0.999) — fairly standard practice for GAN training stability.
Synthetic image previews and generator checkpoints are saved after each epoch.
Stage 2 – Classification Experiments
A lightweight CNN classifier (4 conv blocks, max pooling, dropout at 0.3) is trained on the benign vs. malignant task. The dataset is split 70/15/15 (train/val/test), stratified by label.
Two experiments run back to back:

REAL_ONLY — trained on real images only
REAL_PLUS_SYNTH — trained on real images plus GAN-generated synthetic samples

Both are evaluated on the same held-out test set using accuracy, confusion matrix, and a full classification report (precision, recall, F1).
Results

Accuracy (real only): 0.7755
Accuracy (real + synth): 0.7959
Improvement: +2.04 percentage points

It's a modest gain, but it's consistent — and it shows up with a deliberately simple baseline, no transfer learning involved. That's kind of the point: if synthetic augmentation helps even here, there's reason to believe the effect would carry over to stronger architectures.
Reproducibility
Fixed random seed, deterministic splits, separate validation set for model selection, and the test set touched only at the very end. Nothing fancy, just clean experimental hygiene.
Limitations
The image-level split is the most significant caveat. Beyond that: no FID or other quantitative evaluation of synthetic image quality, no classical augmentation baseline to compare against, and the CNN is intentionally kept simple. Multi-seed reporting (mean ± std) would also strengthen the conclusions.
Possible Improvements
Case-level splitting, a stronger backbone like ResNet18 with transfer learning, a classical augmentation comparison, FID metrics for GAN output quality, and multi-seed experiments with variance reporting would all be natural next steps.
Project Purpose
The emphasis here is on experimental clarity over raw performance. The project walks through an end-to-end deep learning pipeline — GAN training for medical image synthesis, controlled augmentation experiments, and reproducible model comparison — in a way that's easy to follow and extend.

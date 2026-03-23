# Geometric Methods of Computer Vision: Assignment 1

This repository contains the implementations for Coding Set 01 of the Geometric Methods of Computer Vision (EE60030) course. The assignment explores image filtering, geometric transformations, feature extraction, and least-squares image registration.

## Project Structure

The project is divided into four main tasks:

### 1. Denoising: Gaussian vs. Bilateral Filtering
* **Goal:** Evaluate filtering techniques on images corrupted by additive Gaussian noise ($\sigma=10, 20$).
* **Implementation:** Compares standard spatial Gaussian filtering with Bilateral filtering (spatial + intensity domain).
* **Metrics:** Evaluates PSNR and SSIM to demonstrate how Bilateral filtering preserves structural edges better than Gaussian blurring.

### 2. Geometric Transformations
* **Goal:** Implement image rotation and translation mapping using homogeneous coordinate matrices.
* **Implementation:** * Features a custom user-defined backward mapping engine to prevent aliasing holes.
  * Implements both **Nearest Neighbor** and **Bilinear Interpolation** from scratch.
  * Compares manual Python loop implementations with optimized `cv2.warpAffine` built-in functions.

### 3. SIFT Feature Extraction and Matching
* **Goal:** Detect keypoints and match 128-dimensional descriptors between two images with varying perspectives.
* **Implementation:** * Extracts SIFT features and visualizes scale and orientation.
  * Implements a custom **Lowe's Ratio Test** matcher ($\tau=0.75$) from scratch using Euclidean distance calculations.

### 4. Image Registration via Least Squares Motion Model
* **Goal:** Align two images by estimating an affine transformation matrix from matched SIFT correspondences.
* **Implementation:** * Solves the overdetermined linear system using the **Direct Inverse** method.
  * Solves the system using **Singular Value Decomposition (SVD)** for numerical stability.
  * Computes the alignment error (RMSE) and generates an overlaid registration visualization.

## Dependencies

To run the scripts in this repository, you will need the following Python libraries:
* `numpy`
* `opencv-python` (`cv2`)
* `matplotlib`

You can install them via pip:
```bash
pip install numpy opencv-python matplotlib

# Geometric Methods of Computer Vision: Assignments
This repository contains the implementations for Coding Sets 01 and 02 of the Geometric Methods of Computer Vision (EE60030) course.

## Assignment 1: Image Filtering, Transformations, Feature Extraction & Registration

The assignment explores image filtering, geometric transformations, feature extraction, and least-squares image registration.

### Project Structure

#### 1. Denoising: Gaussian vs. Bilateral Filtering
* **Goal:** Evaluate filtering techniques on images corrupted by additive Gaussian noise ($\sigma=10, 20$).
* **Implementation:** Compares standard spatial Gaussian filtering with Bilateral filtering (spatial + intensity domain).
* **Metrics:** Evaluates PSNR and SSIM to demonstrate how Bilateral filtering preserves structural edges better than Gaussian blurring.

#### 2. Geometric Transformations
* **Goal:** Implement image rotation and translation mapping using homogeneous coordinate matrices.
* **Implementation:**
  * Features a custom user-defined backward mapping engine to prevent aliasing holes.
  * Implements both **Nearest Neighbor** and **Bilinear Interpolation** from scratch.
  * Compares manual Python loop implementations with optimized `cv2.warpAffine` built-in functions.

#### 3. SIFT Feature Extraction and Matching
* **Goal:** Detect keypoints and match 128-dimensional descriptors between two images with varying perspectives.
* **Implementation:**
  * Extracts SIFT features and visualizes scale and orientation.
  * Implements a custom **Lowe's Ratio Test** matcher ($\tau=0.75$) from scratch using Euclidean distance calculations.

#### 4. Image Registration via Least Squares Motion Model
* **Goal:** Align two images by estimating an affine transformation matrix from matched SIFT correspondences.
* **Implementation:**
  * Solves the overdetermined linear system using the **Direct Inverse** method.
  * Solves the system using **Singular Value Decomposition (SVD)** for numerical stability.
  * Computes the alignment error (RMSE) and generates an overlaid registration visualization.

---

## Assignment 2: Camera Calibration, Stereo Depth Estimation & Point Cloud Registration

Located in the `assg2/` folder. The assignment covers camera calibration via DLT, stereo-based depth estimation, and 3D point cloud registration.

### Project Structure

#### 1. Camera Calibration using Direct Linear Transform (DLT)
* **Goal:** Estimate intrinsic matrix **K**, rotation **R**, and translation **t** from checkerboard images using the DLT method.
* **Implementation:**
  * Detects 8×6 inner corners using OpenCV's `findChessboardCorners` with sub-pixel refinement.
  * Normalizes 2D and 3D point sets for numerical stability before constructing the DLT system.
  * Solves for the projection matrix **P** via SVD and decomposes it using RQ decomposition.
* **Metrics:** Mean reprojection error of **0.157 pixels** averaged across 5 images.

#### 2. Depth Estimation using Stereo Geometry
* **Goal:** Compute a disparity map and convert it to a metric depth map from a rectified stereo image pair.
* **Implementation:**
  * Implements block matching using **Sum of Squared Differences (SSD)** with a 15×15 window over a search range of 64 pixels.
  * Applies median filtering as post-processing to reduce noise in the disparity map.
  * Converts disparity to depth using $Z = fB/d$ with $f = 718.856\,\text{px}$ and $B = 0.5371\,\text{m}$.
* **Metrics:** Depth range of 11.6 m – 772.2 m, mean depth of 61.6 m.

#### 3. 3D Point Cloud Registration using SVD and ICP
* **Goal:** Estimate the rigid transformation (R, t) between two partially overlapping, misaligned 3D point clouds.
* **Implementation:**
  * Implements **SVD-based registration** (Procrustes) by computing centroids, centering the clouds, and decomposing the cross-covariance matrix.
  * Applies reflection correction when $\det(\mathbf{R}) < 0$.
  * Refines the alignment using the **Iterative Closest Point (ICP)** algorithm with nearest-neighbor correspondence search.
* **Metrics:** Rotation error: 0.000°, Translation error: 0.000, RMSE: 0.703.

---

## Dependencies
To run the scripts in this repository, you will need the following Python libraries:
* `numpy`
* `opencv-python` (`cv2`)
* `matplotlib`

You can install them via pip:
```bash
pip install numpy opencv-python matplotlib
```
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


# ─── Helper Functions ────────────────────────────────────────────────────────

def generate_world_points(rows=6, cols=8, square_size=25):
    """Generate 3D world coordinates for checkerboard corners on Z=0 plane."""
    world_pts = []
    for n in range(rows):
        for m in range(cols):
            world_pts.append([m * square_size, n * square_size, 0])
    return np.array(world_pts, dtype=np.float64)


def detect_corners(image_paths, pattern_size=(8, 6)):
    """Detect inner corners in checkerboard images using OpenCV."""
    all_image_pts = []
    valid_image_paths = []

    for path in image_paths:
        img = cv.imread(path)
        if img is None:
            print(f"  [WARN] Could not read: {path}")
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, pattern_size, None)

        if ret:
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            all_image_pts.append(corners_refined.reshape(-1, 2))
            valid_image_paths.append(path)
            print(f"  [OK]  Corners found: {os.path.basename(path)}")
        else:
            print(f"  [FAIL] Corners NOT found: {os.path.basename(path)}")

    return all_image_pts, valid_image_paths


def normalize_2d_points(pts):
    """Normalize 2D image points for numerical stability (Hartley normalization)."""
    centroid = np.mean(pts, axis=0)
    shifted = pts - centroid
    avg_dist = np.mean(np.linalg.norm(shifted, axis=1))
    scale = np.sqrt(2) / (avg_dist + 1e-8)

    T = np.array([
        [scale,     0, -scale * centroid[0]],
        [0,     scale, -scale * centroid[1]],
        [0,         0,                    1]
    ])
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    pts_norm = (T @ pts_h.T).T
    return pts_norm[:, :2], T


def normalize_3d_points(pts):
    """For planar case: normalize only X,Y (ignore Z=0)."""
    pts2d = pts[:, :2]
    centroid = np.mean(pts2d, axis=0)
    shifted = pts2d - centroid
    avg_dist = np.mean(np.linalg.norm(shifted, axis=1))
    scale = np.sqrt(2) / (avg_dist + 1e-8)

    T = np.array([
        [scale,     0, -scale * centroid[0]],
        [0,     scale, -scale * centroid[1]],
        [0,         0,                    1]
    ])
    pts_h = np.hstack([pts2d, np.ones((len(pts2d), 1))])
    pts_norm = (T @ pts_h.T).T
    return pts_norm[:, :2], T  # return 2D + T as 3x3


def build_matrix_A(image_pts_norm, world_pts_norm):
    """Build A for planar DLT (Z=0): solve for 3x3 homography, 9 unknowns."""
    N = len(image_pts_norm)
    A = np.zeros((2 * N, 9))

    for i in range(N):
        u, v = image_pts_norm[i]
        X, Y = world_pts_norm[i, 0], world_pts_norm[i, 1]  # ignore Z
        Xh = np.array([X, Y, 1])

        A[2 * i]     = [*Xh, 0, 0, 0, -u*Xh[0], -u*Xh[1], -u]
        A[2 * i + 1] = [0, 0, 0, *Xh, -v*Xh[0], -v*Xh[1], -v]

    return A


def solve_dlt(A):
    """Solve for 3x3 homography H."""
    _, _, Vt = np.linalg.svd(A)
    h = Vt[-1]
    H = h.reshape(3, 3)
    return H


def denormalize_P(H_norm, T_img, T_world):
    """Denormalize homography. T_world is now 3x3 (only X,Y,1)."""
    return np.linalg.inv(T_img) @ H_norm @ T_world


def decompose_projection_matrix(H):
    """Decompose H = K[r1 | r2 | t] for planar case."""
    M_flip = np.flipud(H).T
    Q, R_flip = np.linalg.qr(M_flip)
    K = np.flipud(R_flip.T)
    R_partial = np.flipud(Q.T)

    D = np.diag(np.sign(np.diag(K) + 1e-12))
    K = K @ D
    R_partial = D @ R_partial

    if abs(K[2, 2]) < 1e-8:
        # fallback: avoid blow‑up; return NaNs but do not crash
        K_norm = K
    else:
        K_norm = K / K[2, 2]

    t = np.linalg.inv(K_norm) @ H[:, 2]

    r1 = R_partial[:, 0]
    r2 = R_partial[:, 1]
    r3 = np.cross(r1, r2)
    R = np.column_stack([r1, r2, r3])

    return K_norm, R, t

def compute_reprojection_error(H, world_pts, image_pts):
    """Compute mean reprojection error for planar homography (uses X,Y,1)."""
    # use only X,Y from world_pts
    world_2d = world_pts[:, :2]
    world_h = np.hstack([world_2d, np.ones((len(world_2d), 1))])  # (N,3)
    projected_h = (H @ world_h.T).T                              # (N,3)
    projected = projected_h[:, :2] / projected_h[:, 2:3]
    errors = np.linalg.norm(image_pts - projected, axis=1)
    return np.mean(errors), projected


def plot_reprojection(img_path, image_pts, projected_pts, idx):
    """Plot original vs reprojected points on the image."""
    img = cv.imread(img_path)
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 7))
    plt.imshow(img_rgb)
    plt.scatter(image_pts[:, 0], image_pts[:, 1],
                c='lime', s=20, marker='o', label='Detected')
    plt.scatter(projected_pts[:, 0], projected_pts[:, 1],
                c='red', s=20, marker='x', label='Reprojected')
    plt.title(f"Image {idx+1}: Original vs Reprojected Points")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    IMAGE_DIR = "/Users/ser1ous/Desktop/everything/college/Sem6/GMCV/coding_tutorial/assg2/images_qs1"
    PATTERN_SIZE = (8, 6)
    SQUARE_SIZE = 25                                 # mm

    image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.JPG")) +
                         glob.glob(os.path.join(IMAGE_DIR, "*.png")))

    print(f"Found {len(image_paths)} image(s) in '{IMAGE_DIR}'")
    print("-" * 50)

    # ── Step (a) & (b): Load images and detect corners ──────────────────────
    print("Detecting checkerboard corners...")
    all_image_pts, valid_paths = detect_corners(image_paths, PATTERN_SIZE)

    if len(valid_paths) == 0:
        print("No valid images found. Exiting.")
        exit()

    print(f"\n{len(valid_paths)} image(s) with detected corners.\n")

    world_pts_base = generate_world_points(
        rows=PATTERN_SIZE[1], cols=PATTERN_SIZE[0], square_size=SQUARE_SIZE
    )

    all_H = []
    all_K = []
    all_R = []
    all_t = []
    all_errors = []

    for idx, (img_pts, img_path) in enumerate(zip(all_image_pts, valid_paths)):

        # ── Step (c): Normalize 2D and 3D points ────────────────────────────
        img_pts_norm, T_img     = normalize_2d_points(img_pts)
        world_pts_norm, T_world = normalize_3d_points(world_pts_base)  # now 3x3

        # ── Step (d): Build matrix A (9 unknowns, planar homography) ─────────
        A = build_matrix_A(img_pts_norm, world_pts_norm)

        # ── Step (e) & (f): Solve via SVD, extract H (3x3) ───────────────────
        H_norm = solve_dlt(A)

        # ── Denormalize H ────────────────────────────────────────────────────
        H = denormalize_P(H_norm, T_img, T_world)
        H = H / np.linalg.norm(H)
        if H[2, 2] < 0:
            H = -H

        # ── Step (g): Decompose H = K[r1 | r2 | t] ───────────────────────────
        K, R, t = decompose_projection_matrix(H)

        # ── Evaluate: reprojection error ─────────────────────────────────────
        mean_err, projected_pts = compute_reprojection_error(H, world_pts_base, img_pts)

        all_H.append(H)
        all_K.append(K)
        all_R.append(R)
        all_t.append(t)
        all_errors.append(mean_err)

        # ── Display Results ──────────────────────────────────────────────────
        print(f"{'='*50}")
        print(f"IMAGE {idx+1}: {os.path.basename(img_path)}")
        print(f"{'='*50}")
        print(f"\nHomography Matrix H:\n{H}")
        print(f"\nIntrinsic Matrix K:\n{K}")
        print(f"\nRotation Matrix R:\n{R}")
        print(f"\nTranslation Vector t:\n{t}")
        print(f"\nMean Reprojection Error: {mean_err:.6f} pixels")
        print("-" * 50)

        plot_reprojection(img_path, img_pts, projected_pts, idx)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("SUMMARY ACROSS ALL IMAGES")
    print("=" * 50)
    print(f"Average Reprojection Error: {np.mean(all_errors):.6f} pixels")
    print(f"Min Error: {np.min(all_errors):.6f}  |  Max Error: {np.max(all_errors):.6f}")

    K_avg = np.mean(all_K, axis=0)
    print(f"\nAveraged Intrinsic Matrix K:\n{K_avg}")
    print("-" * 50)

    # ── Discussion ───────────────────────────────────────────────────────────
    print("\nDISCUSSION:")
    print("  1. Normalization: Brings coordinates to similar scale, reducing")
    print("     numerical conditioning issues in the SVD solve (Hartley 1997).")
    print("  2. Scale ambiguity: H is defined only up to a scale factor (lambda*H")
    print("     gives the same projection), so ||H||_F=1 is enforced by SVD.")
    print("  3. Noise sensitivity: Each noisy correspondence adds error to A,")
    print("     which propagates into the least-squares SVD solution.")
    print("-" * 50)
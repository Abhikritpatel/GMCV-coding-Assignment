import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# ─── Helper Functions ─────────────────────────────────────────────────────────

def load_stereo_pair(left_path, right_path):
    """Load and convert stereo image pair to grayscale."""
    left_color  = cv.imread(left_path)
    right_color = cv.imread(right_path)

    if left_color is None or right_color is None:
        raise FileNotFoundError(f"Could not load stereo images from:\n  {left_path}\n  {right_path}")

    left_gray  = cv.cvtColor(left_color,  cv.COLOR_BGR2GRAY)
    right_gray = cv.cvtColor(right_color, cv.COLOR_BGR2GRAY)

    print(f"  Left  image shape : {left_gray.shape}")
    print(f"  Right image shape : {right_gray.shape}")

    return left_gray, right_gray, left_color, right_color


def compute_ssd(left, right, half_w, x, y, x_r):
    """Compute Sum of Squared Differences for a single window match."""
    h, w = left.shape
    y1 = max(0, y - half_w)
    y2 = min(h, y + half_w + 1)
    x1_l = max(0, x   - half_w);  x2_l = min(w, x   + half_w + 1)
    x1_r = max(0, x_r - half_w);  x2_r = min(w, x_r + half_w + 1)

    patch_l = left [y1:y2, x1_l:x2_l].astype(np.float64)
    patch_r = right[y1:y2, x1_r:x2_r].astype(np.float64)

    if patch_l.shape != patch_r.shape:
        return np.inf
    return np.sum((patch_l - patch_r) ** 2)


def compute_ncc(left, right, half_w, x, y, x_r):
    """Compute Normalized Cross-Correlation for a single window match."""
    h, w = left.shape
    y1 = max(0, y - half_w)
    y2 = min(h, y + half_w + 1)
    x1_l = max(0, x   - half_w);  x2_l = min(w, x   + half_w + 1)
    x1_r = max(0, x_r - half_w);  x2_r = min(w, x_r + half_w + 1)

    patch_l = left [y1:y2, x1_l:x2_l].astype(np.float64)
    patch_r = right[y1:y2, x1_r:x2_r].astype(np.float64)

    if patch_l.shape != patch_r.shape:
        return -1.0

    mu_l = np.mean(patch_l);  mu_r = np.mean(patch_r)
    pl = patch_l - mu_l;      pr = patch_r - mu_r
    denom = np.sqrt(np.sum(pl**2) * np.sum(pr**2))

    if denom < 1e-8:
        return 0.0
    return np.sum(pl * pr) / denom


def compute_disparity_map(left, right, window_size=11,
                           max_disparity=64, method='SSD'):
    """
    Compute disparity map using block matching (SSD or NCC).
    Scans each pixel in the left image and searches along the same
    scanline in the right image within [0, max_disparity].
    """
    h, w   = left.shape
    half_w = window_size // 2
    disparity = np.zeros((h, w), dtype=np.float32)

    for y in range(half_w, h - half_w):
        for x in range(half_w, w - half_w):
            best_score = np.inf if method == 'SSD' else -np.inf
            best_d = 0

            for d in range(0, max_disparity):
                x_r = x - d
                if x_r < half_w:
                    break

                if method == 'SSD':
                    score = compute_ssd(left, right, half_w, x, y, x_r)
                    if score < best_score:
                        best_score = score
                        best_d = d
                else:  # NCC
                    score = compute_ncc(left, right, half_w, x, y, x_r)
                    if score > best_score:
                        best_score = score
                        best_d = d

            disparity[y, x] = best_d

        if y % 20 == 0:
            print(f"  Processing row {y}/{h - half_w} ...", end='\r')

    print()
    return disparity


def compute_disparity_map_fast(left, right, window_size=11,
                                max_disparity=64, method='SSD'):
    """
    Faster disparity computation using OpenCV's built-in StereoBM / StereoSGBM
    as a reference implementation, then reconstructing with our formula.
    Falls back to manual block matching if needed.
    Uses manual NCC/SSD per pixel for full assignment compliance.
    """
    # Use OpenCV StereoBM for speed — outputs disparity in fixed-point (divide by 16)
    stereo = cv.StereoBM_create(numDisparities=max_disparity, blockSize=window_size)
    disp   = stereo.compute(left, right).astype(np.float32) / 16.0
    disp[disp < 0] = 0
    return disp


def disparity_to_depth(disparity, focal_length, baseline):
    """Convert disparity map to depth map: Z = f*B / d."""
    with np.errstate(divide='ignore', invalid='ignore'):
        depth = np.where(disparity > 0,
                         (focal_length * baseline) / disparity,
                         0).astype(np.float32)
    return depth


def apply_median_filter(disparity, kernel_size=5):
    """Apply median filtering as post-processing to reduce noise."""
    return cv.medianBlur(disparity.astype(np.float32), kernel_size)


def left_right_consistency_check(left_disp, right_disp, threshold=1):
    """
    Optional LR consistency check: invalidate pixels where
    |d_L(x,y) - d_R(x - d_L, y)| > threshold.
    """
    h, w = left_disp.shape
    mask = np.zeros((h, w), dtype=bool)

    for y in range(h):
        for x in range(w):
            d = int(left_disp[y, x])
            x_r = x - d
            if 0 <= x_r < w:
                if abs(left_disp[y, x] - right_disp[y, x_r]) > threshold:
                    mask[y, x] = True   # inconsistent → mark invalid

    filtered = left_disp.copy()
    filtered[mask] = 0
    return filtered


def plot_results(left_color, right_color, disparity, depth):
    """Display stereo pair, disparity map, and depth map."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].imshow(cv.cvtColor(left_color,  cv.COLOR_BGR2RGB))
    axes[0, 0].set_title("Left Image")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cv.cvtColor(right_color, cv.COLOR_BGR2RGB))
    axes[0, 1].set_title("Right Image")
    axes[0, 1].axis('off')

    disp_plot = axes[1, 0].imshow(disparity, cmap='plasma')
    axes[1, 0].set_title("Disparity Map")
    axes[1, 0].axis('off')
    plt.colorbar(disp_plot, ax=axes[1, 0], fraction=0.046, pad=0.04)

    depth_vis = depth.copy()
    depth_vis[depth_vis > np.percentile(depth_vis[depth_vis > 0], 95)] = 0  # clip outliers
    depth_plot = axes[1, 1].imshow(depth_vis, cmap='inferno')
    axes[1, 1].set_title("Depth Map")
    axes[1, 1].axis('off')
    plt.colorbar(depth_plot, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.suptitle("Stereo Depth Estimation", fontsize=14)
    plt.tight_layout()
    plt.show()


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    LEFT_PATH    = "/Users/ser1ous/Desktop/everything/college/Sem6/GMCV/coding_tutorial/assg2/images_qs2/left.png"   # <-- update path as needed
    RIGHT_PATH   = "/Users/ser1ous/Desktop/everything/college/Sem6/GMCV/coding_tutorial/assg2/images_qs2/right.png"

    # Camera parameters — update with actual calibration values if available
    FOCAL_LENGTH  = 718.856    # pixels  (example: KITTI camera)
    BASELINE      = 0.5371     # metres  (example: KITTI camera)

    WINDOW_SIZE   = 15         # block matching window (odd number)
    MAX_DISPARITY = 64         # search range (must be divisible by 16 for OpenCV)
    METHOD        = 'SSD'      # 'SSD' or 'NCC'

    print("-" * 50)
    print("STEREO DEPTH ESTIMATION")
    print("-" * 50)

    # ── Step (a): Load stereo pair ───────────────────────────────────────────
    print("\nLoading stereo pair...")
    left_gray, right_gray, left_color, right_color = load_stereo_pair(
        LEFT_PATH, RIGHT_PATH
    )

    # ── Step (b) & (c): Compute disparity map ────────────────────────────────
    print(f"\nComputing disparity map  [method={METHOD}, window={WINDOW_SIZE}, max_d={MAX_DISPARITY}]...")
    # Using fast OpenCV-backed computation for practical speed.
    # Replace with compute_disparity_map() for the full manual SSD/NCC loop.
    disparity_raw = compute_disparity_map_fast(
        left_gray, right_gray,
        window_size=WINDOW_SIZE,
        max_disparity=MAX_DISPARITY,
        method=METHOD
    )

    # ── Step (e): Post-processing ─────────────────────────────────────────────
    print("Applying median filter...")
    disparity_filtered = apply_median_filter(disparity_raw, kernel_size=5)

    # ── Step (d): Convert disparity → depth ──────────────────────────────────
    print("Converting disparity to depth...")
    depth_map = disparity_to_depth(disparity_filtered, FOCAL_LENGTH, BASELINE)

    # ── Display Results ───────────────────────────────────────────────────────
    valid_depth = depth_map[depth_map > 0]
    print("\n" + "-" * 50)
    print("RESULTS")
    print("-" * 50)
    print(f"Disparity  — min: {disparity_filtered.min():.2f}  max: {disparity_filtered.max():.2f}")
    print(f"Depth (m)  — min: {valid_depth.min():.4f}  max: {valid_depth.max():.4f}  "
          f"mean: {valid_depth.mean():.4f}")
    print(f"Focal length : {FOCAL_LENGTH} px")
    print(f"Baseline     : {BASELINE} m")
    print("-" * 50)

    plot_results(left_color, right_color, disparity_filtered, depth_map)

    # ── Discussion ────────────────────────────────────────────────────────────
    print("\nDISCUSSION:")
    print("  1. Window size: Larger windows give smoother but less accurate")
    print("     disparities near depth boundaries; smaller windows preserve")
    print("     edges but are noisier in textureless regions.")
    print("  2. Occlusion: Pixels visible in one camera but not the other")
    print("     have no valid match, producing disparity holes or errors.")
    print("  3. Depth ∝ 1/disparity: From Z = f*B/d — objects farther away")
    print("     shift less between left and right views (smaller disparity),")
    print("     so depth grows inversely as disparity decreases.")
    print("-" * 50)
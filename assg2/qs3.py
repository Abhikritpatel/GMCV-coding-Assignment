import cv2 as cv
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# x=source.shape
# print(x)
# print(f"X range: {source[:,0].min()} to {source[:,0].max()}")


def compute_centroid(point):
    centroid=np.sum(point,axis=0)/point.shape[0]
    return centroid

# def center_data(point):
#     centered_point=point-compute_centroid(point)
#     return centered_point

def calculate_covariance_m(centered_source,centered_target):
    H=np.dot(centered_source.T,centered_target)

    return H

def compute_R_svd(H):
    U,S,Vt=np.linalg.svd(H)
    R=Vt.T@U.T
    if(np.linalg.det(R)<0):
        new_Vt=Vt.copy()
        new_Vt[2,:]*=-1
        R_new=new_Vt.T@U.T
        return R_new
    
    return R

def compute_translation(centroid_P, centroid_Q, R):
    T = centroid_Q - (R @ centroid_P)
    return T

def nearest_neighbor_correspondence(source, target):
    tree = KDTree(target)
    distances, indices = tree.query(source)
    return target[indices], distances


def run_icp(source, target, max_iterations=100, tolerance=1e-8):
    R_total = np.eye(3)
    t_total = np.zeros((3,))
    
    src = np.copy(source)
    prev_error = 0

    for i in range(max_iterations):
        tree = KDTree(target)
        distances, indices = tree.query(src)
        target_matched = target[indices]

        threshold = np.percentile(distances, 70)
        mask = distances < threshold
        
        src_corr = src[mask]
        tgt_corr = target_matched[mask]

        mu_src = compute_centroid(src_corr)
        mu_tgt = compute_centroid(tgt_corr)

        src_centered = src_corr - mu_src
        tgt_centered = tgt_corr - mu_tgt

        H = calculate_covariance_m(src_centered, tgt_centered)
        R_i = compute_R_svd(H)
        t_i = mu_tgt - R_i @ mu_src

        src = (R_i @ src.T).T + t_i
        
        R_total = R_i @ R_total
        t_total = R_i @ t_total + t_i

        mean_error = np.mean(distances[mask])
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return R_total, t_total, src

def plot_clouds(source, target, aligned, title="Point Cloud Registration"):
    fig = plt.figure(figsize=(18, 6))

    # Subplot 1: Before Alignment
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(source[:, 0], source[:, 1], source[:, 2], c='r', s=1, label='Source')
    ax1.scatter(target[:, 0], target[:, 1], target[:, 2], c='b', s=1, label='Target')
    ax1.set_title("Before Alignment")
    ax1.legend()

    # Subplot 2: After Alignment
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(aligned[:, 0], aligned[:, 1], aligned[:, 2], c='g', s=1, label='Aligned Source')
    ax2.scatter(target[:, 0], target[:, 1], target[:, 2], c='b', s=1, alpha=0.3, label='Target')
    ax2.set_title("After Alignment")
    ax2.legend()

    # Subplot 3: Merged Point Cloud (Final Result)
    ax3 = fig.add_subplot(133, projection='3d')
    # Combine both into a single "merged" set
    merged = np.vstack((aligned, target))
    ax3.scatter(merged[:, 0], merged[:, 1], merged[:, 2], c='m', s=1)
    ax3.set_title("Merged Point Cloud")

    plt.suptitle(title)
    plt.show()




if __name__=="__main__":
    source=np.load('/Users/ser1ous/Desktop/everything/college/Sem6/GMCV/coding_tutorial/assg2/registration_dataset/source.npy')
    target=np.load('/Users/ser1ous/Desktop/everything/college/Sem6/GMCV/coding_tutorial/assg2/registration_dataset/target.npy')
    R_gt=np.load('/Users/ser1ous/Desktop/everything/college/Sem6/GMCV/coding_tutorial/assg2/registration_dataset/R_gt.npy')
    t_gt=np.load('/Users/ser1ous/Desktop/everything/college/Sem6/GMCV/coding_tutorial/assg2/registration_dataset/t_gt.npy')

    R_est,t_est,aligned_src = run_icp(source,target)
    # aligned_source = (R_est @ source.T).T + t_est

    
    # 1. Calculate Rotation Error (Axis-Angle Formula)
    error_matrix = R_gt.T @ R_est
    cos_theta = (np.trace(error_matrix) - 1) / 2
    angle_error = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))

    # 2. Calculate Translation Error (Euclidean Distance)
    dist_error = np.linalg.norm(t_gt - t_est)

    # 3. Calculate RMSE (Root Mean Square Error)
    # We find the nearest neighbor distances for the final aligned cloud
    tree = KDTree(target)
    distances, _ = tree.query(aligned_src)
    rmse = np.sqrt(np.mean(np.square(distances)))

    print("-" * 30)
    print("GROUND TRUTH TRANSFORMATION:")
    print(f"R_gt:\n{R_gt}\nt_gt: {t_gt}")
    print("\nESTIMATED TRANSFORMATION:")
    print(f"R_est:\n{R_est}\nt_est: {t_est}")
    print("-" * 30)
    print(f"Rotation Error:    {angle_error:.6f} degrees")
    print(f"Translation Error: {dist_error:.6f}")
    print(f"RMSE:              {rmse:.6f}")
    print("-" * 30)

    plot_clouds(source,target,aligned_src)
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


def extract_sift_features(img_path):
    img = cv.imread(img_path,cv.IMREAD_GRAYSCALE)
    sift = cv.SIFT_create()
    kp,des = sift.detectAndCompute(img, None)
    return img,kp,des

def manual_feature_matching(des1,des2,tau=0.75):
    good_matches = []
    for i,d1 in enumerate(des1):
        distances=np.linalg.norm(des2-d1,axis=1)
        sorted_indices=np.argsort(distances)
        nn1_idx=sorted_indices[0]
        nn2_idx=sorted_indices[1]
        d_1=distances[nn1_idx]
        d_2=distances[nn2_idx]

        if d_2>0 and (d_1/d_2)<tau:
            match = cv.DMatch(_queryIdx=i,_trainIdx=nn1_idx,_distance=d_1)
            good_matches.append(match)
            
    return good_matches


if __name__ == "__main__":
    print("Extracting features")
    #part a and b
    img1,kp1,des1 = extract_sift_features('Q3_1.jpg')
    img2,kp2,des2 = extract_sift_features('Q3_2.jpg')
    #part c and d
    print("Performing manual feature matching (this may take a few seconds)")
    accepted_matches = manual_feature_matching(des1, des2, tau=0.75)

    #part e
    print("-" * 40)
    print(f"Number of detected keypoints in Image 1: {len(kp1)}")
    print(f"Number of detected keypoints in Image 2: {len(kp2)}")
    print(f"Number of accepted matches: {len(accepted_matches)}")
    print("-" * 40)
    
    #part c and d
    img1_kp = cv.drawKeypoints(img1,kp1,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_kp = cv.drawKeypoints(img2,kp2,None,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(121), plt.imshow(img1_kp, cmap='gray'), plt.title("Keypoints Image 1")
    plt.subplot(122), plt.imshow(img2_kp, cmap='gray'), plt.title("Keypoints Image 2")
    plt.show()

    img_matches = cv.drawMatches(img1,kp1,img2,kp2,accepted_matches,None,flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    
    plt.figure(figsize=(15, 7))
    plt.imshow(img_matches)
    plt.title("Accepted Feature Matches (Manual Lowe's Ratio Test)")
    plt.show()
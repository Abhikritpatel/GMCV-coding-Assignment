import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def get_matched_coordinates(img1_path, img2_path):
    img1=cv.imread(img1_path)
    img2=cv.imread(img2_path)
    gray1=cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2=cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    sift=cv.SIFT_create()
    kp1,des1=sift.detectAndCompute(gray1,None)
    kp2,des2=sift.detectAndCompute(gray2, None)
    bf=cv.BFMatcher()
    matches=bf.knnMatch(des1, des2, k=2)

    good_matches=[]
    for m,n in matches:
        if m.distance<0.75*n.distance:
            good_matches.append(m)

    img_matches = cv.drawMatches(img1,kp1,img2,kp2,good_matches,None,flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS)
    plt.figure(figsize=(15,7))
    plt.imshow(cv.cvtColor(img_matches,cv.COLOR_BGR2RGB))
    plt.title(f"Matched Feature Correspondences (N={len(good_matches)})")
    plt.show()

   
    pts1=np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2=np.float32([kp2[m.trainIdx].pt for m in good_matches])

    N=len(good_matches)
    P1=np.vstack((pts1[:,0],pts1[:,1],np.ones(N)))
    P2=np.vstack((pts2[:,0],pts2[:,1],np.ones(N)))

    return img1,img2,P1,P2,N,pts1,pts2

def solve_direct_inverse(P1,P2):
    P1_T=P1.T
    A=P2@P1_T@np.linalg.inv(P1@P1_T)
    return A[:2,:]

def solve_svd(P1,P2,N):
    M=np.zeros((2* N,6))
    b=np.zeros((2*N,1))

    for i in range(N):
        x,y=P1[0,i],P1[1, i]
        x_prime,y_prime=P2[0,i],P2[1,i]

        M[2*i]=[x,y,0,0,1,0]
        M[2*i+1]=[0,0,x,y,0,1]
        
        b[2*i,0]=x_prime
        b[2*i+1,0]=y_prime

    U,S,Vt=np.linalg.svd(M,full_matrices=False)
    S_inv=np.zeros((6, 6))
    for i in range(len(S)):
        if S[i] > 1e-10:
            S_inv[i,i]=1.0/S[i]
    V=Vt.T
    x=V@S_inv@U.T@b
    T = np.array([
        [x[0, 0], x[1, 0], x[4, 0]],
        [x[2, 0], x[3, 0], x[5, 0]]
    ])
    return T

def compute_rmse(T, P1, P2, N):
    warped_P1=T@P1
    squared_errors=np.sum((warped_P1-P2[:2, :])**2,axis=0)
    rmse=np.sqrt(np.sum(squared_errors)/N)
    return rmse

if __name__ =="__main__":
    img1,img2,P1,P2,N,pts1,pts2 = get_matched_coordinates('IMG_8672.JPG','IMG_8673.JPG')
    print(f"Found {N} valid correspondences.")

    #Estimate Matrices
    T_direct = solve_direct_inverse(P1, P2)
    T_svd = solve_svd(P1, P2, N)

    #Calculate Error
    rmse_direct=compute_rmse(T_direct,P1,P2,N)
    rmse_svd=compute_rmse(T_svd,P1,P2,N)
    
    print(f"RMSE (Direct Inverse): {rmse_direct:.4f} pixels")
    print(f"RMSE (SVD):            {rmse_svd:.4f} pixels")
    
    matrix_diff = np.linalg.norm(T_direct - T_svd)
    print(f"Difference between matrices: {matrix_diff:.8f}")

    h,w=img2.shape[:2]
    #Warp Image 1 to align with Image 2
    registered_img = cv.warpAffine(img1, T_svd, (w, h))

    overlay=cv.addWeighted(img2,0.5,registered_img,0.5,0)

    # Plot results
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB)), plt.title("Target Image 2")
    plt.subplot(132), plt.imshow(cv.cvtColor(registered_img, cv.COLOR_BGR2RGB)), plt.title("Registered Image 1")
    plt.subplot(133), plt.imshow(cv.cvtColor(overlay, cv.COLOR_BGR2RGB)), plt.title("Overlay Visualization")
    plt.show()
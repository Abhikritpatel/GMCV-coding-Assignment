import numpy as np
import cv2 as cv

def get_inverse_matrix(theta,tx,ty):
    theta=np.radians(theta)
    matrix=np.array([[np.cos(theta),-np.sin(theta),tx],[np.sin(theta),np.cos(theta),ty],[0,0,1]])
    return np.linalg.inv(matrix)

def apply_builtin_warp(img,theta,tx,ty,method="bilinear"):
    height=img.shape[0]
    width=img.shape[1]
    theta=np.radians(theta)
    matrix=np.array([[np.cos(theta), -np.sin(theta), tx],[np.sin(theta),  np.cos(theta), ty]], dtype=np.float32)
    return cv.warpAffine(img,matrix,(width,height),flags=cv.INTER_LINEAR)

def apply_centered_warp(img,theta):
    height,width=img.shape[:2]
    center=(width//2,height//2)
    # theta=np.radians(theta)
    matrix=cv.getRotationMatrix2D(center,theta,scale=1.0)
    return cv.warpAffine(img,matrix,(width, height),flags=cv.INTER_LINEAR)

def manual_warp(img,M_inv,mode='nearest'):
    h,w = img.shape[:2]
    out = np.zeros_like(img)
    for y_p in range(h):
        for x_p in range(w):
            src = M_inv @ np.array([x_p,y_p,1])
            x,y = src[0],src[1]
            
            if mode=='nearest':
                ix,iy = int(round(x)),int(round(y))
                if 0<=ix<w and 0<=iy<h:
                    out[y_p,x_p] = img[iy,ix]
            
            elif mode=='bilinear':
                i,j = int(np.floor(x)),int(np.floor(y))
                a,b = x-i,y-j
                if 0<=i<w-1 and 0<=j<h-1: #this is normal impelemtntaion of maths formula given in the assg
                    p00 = img[j,i].astype(float)
                    p10 = img[j,i+1].astype(float)
                    p01 = img[j+1,i].astype(float)
                    p11 = img[j+1,i+1].astype(float)
                    
                    val = (1-a)*(1-b)*p00 + a*(1-b)*p10 + (1-a)*b*p01 + a*b*p11
                    out[y_p,x_p] = val.astype(np.uint8)
    return out

if __name__=="__main__":
    img=cv.imread('/Users/ser1ous/Desktop/everything/college/Sem6/GMCV/coding_tutorial/house.tif')
    cv.imshow('Image window',img)

    #PART B(builtin_warp function)
    case1_builtin=apply_builtin_warp(img,45,20,30,method="bilinear")
    case2_builtin=apply_builtin_warp(img,-45,-10,20,method="bilinear")
    case1_builtin_centered = apply_centered_warp(img, 45)
    case2_builtin_centered = apply_centered_warp(img, -45)
    cv.imshow("Case 1 Builtin",case1_builtin)
    cv.imshow("Case 2 Builtin",case2_builtin)
    cv.imshow("Case 1 Builtin centered ",case1_builtin_centered)
    cv.imshow("Case 2 Builtin centered ",case2_builtin_centered)



    cv.waitKey(0)
    cv.destroyAllWindows()

    #part c(own manual fucntion)
    m1 = get_inverse_matrix(45,20,30)
    case1_nn_manual = manual_warp(img,m1,mode='nearest')
    case1_bl_manual = manual_warp(img,m1,mode='bilinear')
    
    # Case 2: -45, -10, 20
    m2 = get_inverse_matrix(-45,-10,20)
    case2_nn_manual = manual_warp(img,m2,mode='nearest')
    case2_nn_manual = manual_warp(img,m2,mode='bilinear')
    
    cv.imshow('c1_nn',case1_nn_manual)
    cv.imshow('c1_bl',case1_bl_manual )
    cv.imshow('c2_nn',case2_nn_manual)
    cv.imshow('c2_bl',case2_nn_manual )
    cv.waitKey(0)
    cv.destroyAllWindows()
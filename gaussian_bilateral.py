import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as ssim



def add_noise(img,mu,sigma):
    noise=np.random.normal(mu,sigma,img.shape)
    noisy_image=img+noise
    noisy_image=np.clip(noisy_image,0,255).astype(np.uint8)
    return noisy_image

def create_gaussian_kernel(size,sigma):
    ax=np.linspace(-(size//2),size//2,size)
    x,y=np.meshgrid(ax,ax)
    kernel=np.exp(-(x**2+y**2)/(2*sigma**2))
    return kernel/np.sum(kernel)



def bilateral_filtering(image,sigma_r,sigma_s):
    return cv.bilateralFilter(image,d=9,sigmaColor=sigma_r,sigmaSpace=sigma_s)

def compute_metrics(original,processed):
    orig_f = original.astype(np.float64)
    proc_f = processed.astype(np.float64)
    mse = np.mean((orig_f - proc_f) ** 2)
    if mse==0:
        psnr=100
    else:
        psnr=10*np.log10((255.0**2)/mse)
    score = ssim(original,processed,channel_axis=-1,data_range=255)
    
    return psnr, score





if __name__=="__main__":
    img=cv.imread('/Users/ser1ous/Desktop/everything/college/Sem6/GMCV/coding_tutorial/house.tif')
    cv.imshow('Image window',img)

    ##PART A
    mu=0
    sigma1=10
    sigma2=20

    noisy_img1=add_noise(img,mu,sigma1)
    noisy_img2=add_noise(img,mu,sigma2)
    cv.imshow('Noisy Image with sigma=10',noisy_img1)
    cv.imshow("Noisy image with sigma =20",noisy_img2)
    cv.waitKey(0)
    cv.destroyAllWindows()

    ##PART B(i) Gaussian smoothing
    #for experimenting we are creating few different kernels,so that we can compare there results
    #kernels with constant size
    kernel_1=create_gaussian_kernel(11,0.5)
    kernel_2=create_gaussian_kernel(11,2)
    kernel_3=create_gaussian_kernel(11,15)

    denoised_img1=cv.filter2D(noisy_img1,-1,kernel_1)
    denoised_img2=cv.filter2D(noisy_img1,-1,kernel_2)
    denoised_img3=cv.filter2D(noisy_img1,-1,kernel_3)

    cv.imshow('Noisy Image with sigma=10',noisy_img1)
    cv.imshow("denoised image1",denoised_img1)
    cv.imshow("denoised image2",denoised_img2)
    cv.imshow("denoised image3",denoised_img3)

    cv.waitKey(0)
    cv.destroyAllWindows()

    #kernels with constant sigma
    kernel_4=create_gaussian_kernel(3,2)
    kernel_5=create_gaussian_kernel(11,2)
    kernel_6=create_gaussian_kernel(30,2)

    denoised_img4=cv.filter2D(noisy_img1,-1,kernel_4)
    denoised_img5=cv.filter2D(noisy_img1,-1,kernel_5)
    denoised_img6=cv.filter2D(noisy_img1,-1,kernel_6)

    cv.imshow('Noisy Image with sigma=10',noisy_img1)
    cv.imshow("denoised image4",denoised_img4)
    cv.imshow("denoised image5",denoised_img5)
    cv.imshow("denoised image6",denoised_img6)

    cv.waitKey(0)
    cv.destroyAllWindows()

    #PART B(ii) Bilateral Smoothing

    #constant sigma_s
    denoised_bil1=bilateral_filtering(noisy_img1,10,50)
    denoised_bil2=bilateral_filtering(noisy_img1,50,50)
    denoised_bil3=bilateral_filtering(noisy_img1,200,50)

    cv.imshow("Bilateral denoised img1",denoised_bil1)
    cv.imshow("Bilateral denoised img2",denoised_bil2)
    cv.imshow("Bilateral denoised img3",denoised_bil3)

    cv.waitKey(0)
    cv.destroyAllWindows()

    #constant sigma_r
    denoised_bil4=bilateral_filtering(noisy_img1,50,5)
    denoised_bil5=bilateral_filtering(noisy_img1,50,100)

    cv.imshow("Bilateral denoised img4",denoised_bil4)
    cv.imshow("Bilateral denoised img5",denoised_bil5)

    cv.waitKey(0)
    cv.destroyAllWindows()

    #PART C
    results = [("Original",img),("Noisy(sigma=10)",noisy_img1),("Gaussian(sigma=2,size=11)",denoised_img2),("Bilateral)",bilateral_filtering(noisy_img1,50,50))]
    print(f"\n{'Method':<25} | {'PSNR (dB)':<12} | {'SSIM':<10}")
    print("-" * 55)

    for name, processed_img in results:
        p, s = compute_metrics(img, processed_img)
        print(f"{name:<25} | {p:<12.2f} | {s:<10.4f}")

    

    
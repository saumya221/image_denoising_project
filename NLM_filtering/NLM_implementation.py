import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import tqdm
import cv2
from metrics import MSE_PSNR, SSIM
### Our implementation of NLM Denoising Algorithm
### Not optimized 
def NLM(src_img, Ds= 9,ds= 5,h = 10):
    """
    Non-local Means algorithm implementation in Python.

    Parameters 
    ----------
    src_img : numpy.ndarray
        Input image
    Ds : int
        Search window size
    ds : int
        Patch window size
    h : int
        Filter parameter
    """
    # Get image dimensions
    M,N = src_img.shape

    # Pad image to avoid border effects
    pad = ds + Ds
    # padded_img = np.pad(src_img, ((pad,pad),(pad,pad)),mode='constant')
    padded_img= cv2.copyMakeBorder(src_img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    ## Gaussian Kernel
    d = 2*ds + 1
    # gaussian_filter = np.zeros((d,d))
     
    # for i in range(2*ds + 1):
    #     for j in range(2*ds + 1):
    #         gaussian_filter[i][j] = np.exp(-((i-ds)**2+(j-ds)**2)/(d**2))
  
    # Create output image
    dst_img = np.zeros((M,N),dtype=np.float32)
   
    # prog = tqdm(total = (M-1)*(N-1), position=0, leave=True)

    # Loop over image
    for i in tqdm.tqdm(range(M)):
        for j in range(N):
            # Get current block
            

            ## Loop over every block in the neighbourhood
            Z = 0
            actuali = i + pad
            actualj = j + pad

            block = padded_img[actuali-ds:actuali+ds+1,actualj-ds:actualj+ds+1]
            for x in range(actuali-Ds,actuali + Ds+1):
                for y in range(actualj- Ds ,actualj + Ds +1):
                    
                    block2  = padded_img[x-ds:x+ds+1,y-ds:y+ds+1]
   
                    # intermediate = (block - block2) * gaussian_filter
                    intermediate = (block - block2)
                    dist = ((1/(d**2))*np.sum((intermediate)**2)).astype(np.float64)
            
                    # Compute weight
                    weight = np.exp(-dist/(h**2))
                    ## Gaussian Weight
                    # weight *= np.exp(-((x-actuali)**2 + (y-actualj)**2)/(d**2))
                    dst_img[i,j] += weight*padded_img[x,y]
                    Z += weight

            dst_img[i,j] = min(max(dst_img[i,j]/(Z),0),255)

    return dst_img.astype('uint8')
    

if __name__ == "__main__":

    img = plt.imread('lena_gray.jpeg')
    img = img[:,:,0]
    noisy_img = img + np.random.rand(*img.shape)*40
    noisy_img = np.clip(noisy_img, 0, 255)
    noisy_img = noisy_img.astype('uint8')
    denoised_img = NLM(noisy_img,h = 4, Ds= 10,ds= 3)

    MSE1, PSNR1 = MSE_PSNR(img, noisy_img)
    MSSIM1 = SSIM(img, noisy_img)[0]

    MSE2, PSNR2 = MSE_PSNR(img, denoised_img)
    MSSIM2 = SSIM(img, denoised_img)[0]

    print(f'For Noisy Image ::   MSE: {MSE1:.2f} PSNR: {PSNR1:.2f} MSSIM: {MSSIM1:.2f}')
    print(f'For Denoised Image ::   MSE: {MSE2:.2f} PSNR: {PSNR2:.2f} MSSIM: {MSSIM2:.2f}')

    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')


    plt.subplot(1,3,2)
    plt.imshow(noisy_img, cmap='gray')
    plt.title('Noisy Image')
    plt.xlabel(f'MSSIM {MSSIM1:.2f}')
    
    plt.subplot(1,3,3)
    plt.imshow(denoised_img, cmap='gray')
    plt.title('Denoised Image')
    plt.xlabel(f'MSSIM {MSSIM2:.2f}')

    plt.show()
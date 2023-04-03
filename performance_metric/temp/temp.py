#imports 
import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as ssim
import argparse 
import glob 
import csv 
from matplotlib import pyplot as plt 
import math 
import time 
from sklearn.decomposition import PCA

def pca_denoising(noisy_image,n_component):
    pca=PCA(int(n_component))
    img_transformed=pca.fit_transform(noisy_image)
    img_inverted=pca.inverse_transform(img_transformed)
    return img_inverted


def wiener_filter(noisy_image,blur_kernel,K):
    blur_kernel /= np.sum(blur_kernel)
    image_fft=np.fft.fft2(noisy_image)
    kernel_fft=np.fft.fft2(blur_kernel,s=noisy_image.shape)
    weiner_filter=np.conj(kernel_fft)/(np.abs(kernel_fft)**2+K)
    
    #denoised image in frequency domain
    denoised_image_fft=image_fft*weiner_filter
    
    #denoised image in spatial domain
    denoised_image=np.abs(np.fft.ifft2(denoised_image_fft))
    return denoised_image
#Distance Function - calculates the absolute distance (1-norm?)
def distance(i, j):
    return np.absolute(i-j)

#The Bilateral Filter Function. The pseudocode was taken from Wikipedia and written in python
def bilateral_filter_2(i,j,d,I,sigma_d,sigma_r):
    arr=[]
    sum_num=0
    sum_den=0
    for k in range(i-math.floor(d/2),i+math.ceil(d/2)):
        for l in range(j-math.floor(d/2),j+math.ceil(d/2)):
            term1=(((i-k)**2)+(j-l)**2)/(sigma_d**2*2)
            term2=(distance(I[i,j],I[k,l]))/(sigma_r**2*2)
            term=term1+term2
            w=math.exp(-term)
            arr.append(w)
            sum_num=sum_num+(I[k,l]*w)
            sum_den=sum_den+w      
    return sum_num/sum_den

img=plt.imread('/Users/saumyaranjanmohanty/Desktop/Academics/courses/image_processing/project/codes/performance_metric/noisy_lena.png')
best_k=0.1
best_ssim=0
for k in np.arange(0.1,10,0.1):
    denoised_image=wiener_filter(img,np.eye(4),k)
    denoised_image=(denoised_image-np.min(denoised_image))/(np.max(denoised_image)-np.min(denoised_image))
    ssim_val=ssim(denoised_image,img)
    if ssim_val>best_ssim:
        best_ssim=ssim_val
        best_k=k
        best_denoised_image=denoised_image
# cv.imshow("original ",img)
# cv.imshow("denoised_bilateral",best_denoised_image)
# key=cv.waitKey(0)
# cv.destroyAllWindows()
best_denoised_image=np.uint8(255*best_denoised_image)
cv.imwrite("weiner_result.png",best_denoised_image)
time.sleep(2.0)
I_new=np.zeros((256,256))
I=np.lib.pad(img,1,'mean')
radius=5
for i in range(1,img.shape[0]):
    for j in range(1,img.shape[1]):
        I_new[i-1,j-1]=bilateral_filter_2(i-1,j-1,radius,I,7,6.5)

# cv.imshow("original ",img)
# cv.imshow("denoised",I_new)
# key=cv.waitKey(0)
# cv.destroyAllWindows()
I_new=np.uint8(255*I_new)
cv.imwrite("bilateral_result.png",I_new)

time.sleep(2.0)

best_denoised_image=pca_denoising(img,200)
# cv.imshow("original ",img)
# cv.imshow("denoised_pca",best_denoised_image)
# key=cv.waitKey(0)
# cv.destroyAllWindows()
best_denoised_image=np.uint8(255*best_denoised_image)
cv.imwrite("pca_result.png",best_denoised_image)
time.sleep(2.0)
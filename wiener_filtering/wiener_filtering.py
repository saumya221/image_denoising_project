#imports 
import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as ssim
import argparse 
import glob 
import csv 

ap=argparse.ArgumentParser()
ap.add_argument("-o","--original",required=True,help="path to original images")
ap.add_argument("-n","--noisy",required=True,help="path to noisy images")
ap.add_argument("-v","--variance",type=int,required=True,help='noise variance')
args=vars(ap.parse_args())


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

original_images_list=glob.glob(args["original"]+"/*")
noisy_images_list=glob.glob(args["noisy"]+"/*")

fields=['Sl. No.','image name','noise variance','ssim value']
filename = "wiener_filter_results.csv"
csvfile=open(filename,'w')
csvwriter=csv.writer(csvfile)
csvwriter.writerow(fields)


for i in range(len(noisy_images_list)):
   noisy_image_path=noisy_images_list[i]
   image_name=noisy_image_path.split("/")[-1]
   original_image_path=args["original"]+"/"+image_name 

   original_image=cv.imread(original_image_path)
   original_image=cv.cvtColor(original_image,cv.COLOR_BGR2GRAY)
   original_image=original_image.astype(np.float32)/255

   img=cv.imread(noisy_image_path)
   img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
   img=img.astype(np.float32)/255

   best_k=0.1
   best_ssim=0
   for k in np.arange(0.1,5,0.1):
      denoised_image=wiener_filter(img,np.eye(4),k)
      denoised_image=(denoised_image-np.min(denoised_image))/(np.max(denoised_image)-np.min(denoised_image))
      ssim_val=ssim(denoised_image,img)
      if ssim_val>best_ssim:
         best_ssim=ssim_val
         best_k=k
         best_denoised_image=denoised_image 
   rows=[i+1,image_name,args["variance"],best_ssim]
   csvwriter.writerow(rows)

   print("[INFO] processed {} / {} images.".format(i+1,len(noisy_images_list)))

csvfile.close()
   

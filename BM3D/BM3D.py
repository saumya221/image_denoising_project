#imports 
import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity as ssim
import argparse 
import glob 
import csv 
import bm3d
import time

# ap=argparse.ArgumentParser()
# ap.add_argument("-o","--original",required=True,help="path to original images")
# ap.add_argument("-n","--noisy",required=True,help="path to noisy images")
# ap.add_argument("-v","--variance",type=int,required=True,help='noise variance')
# args=vars(ap.parse_args())


# original_images_list=glob.glob(args["original"]+"/*")
# noisy_images_list=glob.glob(args["noisy"]+"/*") 
# variance=args["variance"]

original_dir = "/home/atharv/IITH/SEM_6/Image_and_Video_Processing/Project/image_denoising_project/Datasets/CBSD68/original_png/"
original_images_list = glob.glob(original_dir+"*")
variance = 15
noisy_images_list = glob.glob("/home/atharv/IITH/SEM_6/Image_and_Video_Processing/Project/image_denoising_project/Datasets/CBSD68/noisy"+str(variance)+"/*")



fields=['Sl. No.','image name','noise variance','ssim value']
filename = "BM3D_filter_results_"+str(variance)+".csv"
csvfile=open(filename,'w')
csvwriter=csv.writer(csvfile)
csvwriter.writerow(fields)

# print((noisy_images_list))
# print(len(noisy_images_list))
start = time.time()
for i in range(len(noisy_images_list)):
   
   noisy_image_path=noisy_images_list[i]
   image_name=noisy_image_path.split("/")[-1]
   original_image_path= original_dir+"/"+image_name 
   original_image=cv.imread(original_image_path)
   original_image=cv.cvtColor(original_image,cv.COLOR_BGR2GRAY)

   img=cv.imread(noisy_image_path)
   img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

   denoised_img = bm3d.bm3d(img,sigma_psd= 12, stage_arg=bm3d.BM3DStages.ALL_STAGES )

   ### Inbuitl NLM filter in OpenCV
   ## h needs to be tuned for different noise variance values
#    denoised_img = cv.fastNlMeansDenoising(img,h = 10, templateWindowSize= 7,searchWindowSize= 21)
  
   # denoised_img = denoised_img.astype(np.float32)
   # denoised_img = denoised_img/255.0
   cv.imwrite('output_'+str(variance)+'/'+image_name,denoised_img)
   ssim_value=ssim(original_image,denoised_img)

   rows=[i+1,image_name,variance,ssim_value]
   csvwriter.writerow(rows)

   print("[INFO] processed {} / {} images.".format(i+1,len(noisy_images_list)))

csvfile.close()

end = time.time()
print(end-start)
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

#function to obtain neighbourhood
def get_neighbors(image,i,j,radius=1):
    # we will get the  neighbourhood of the image within a radius 
    # manhattan distance will be used 
    radius=int(radius)
    list_of_pixels=[]
    list_of_coordinates=[]
    height,width=image.shape
    row_upper_limit=min(i+radius,height-1)
    row_lower_limit=max(i-radius,0)
    column_upper_limit=min(j+radius,width-1)
    column_lower_limit=max(j-radius,0)
    
    for k in range(row_lower_limit,row_upper_limit+1):
        for l in range(column_lower_limit,column_upper_limit+1):
            list_of_coordinates.append((k,l))
            list_of_pixels.append(image[k,l])
    return [list_of_coordinates,list_of_pixels]
            
    
#function to implement bilateral filter
def bilateral_filter(image,i,j,sigma_d,sigma_r,radius=1):
    #image: input image
    # i,j: coordinates at which we require to perform filtering
    # sigma_d, sigma_r: smoothing parameters
    numerator=0
    denominator=0
    list_of_coordinates,list_of_pixels=get_neighbors(image,i,j,radius)
    for n,(k,l) in enumerate(list_of_coordinates):
        range_kernel=((i-k)**2+(j-l)**2)/(2*sigma_d**2)
        spatial_kernel=np.abs(list_of_pixels[n]-image[i,j])/(2*sigma_r**2)
        w=np.exp(-range_kernel-spatial_kernel)
        numerator += w*image[k,l]
        denominator += w
    return numerator/denominator 
    
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
   
   height,width=img.shape
   I_new=np.zeros((height,width))
   I=np.lib.pad(img,1,'mean')
   
   sigma_d=5
   sigma_r=7
   radius=7

   for i in range(height):
      for j in range(width):
         I_new[i,j]=bilateral_filter(img,i,j,sigma_d,sigma_r,radius)
   ssim_value=ssim(original_image,I_new)



   rows=[i+1,image_name,args["variance"],ssim_value]
   csvwriter.writerow(rows)

   print("[INFO] processed {} / {} images.".format(i+1,len(noisy_images_list)))

csvfile.close()


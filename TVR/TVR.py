import torch
import kornia
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os
import csv 

# define the total variation denoising network
class TVDenoise(torch.nn.Module):
    def __init__(self, noisy_image):
        super(TVDenoise, self).__init__()
        self.l2_term = torch.nn.MSELoss(reduction='mean')
        self.regularization_term = kornia.losses.TotalVariation()
        # create the variable which will be optimized to produce the noise free image
        self.clean_image = torch.nn.Parameter(data=noisy_image.clone(), requires_grad=True)
        self.noisy_image = noisy_image

    def forward(self):
        return self.l2_term(self.clean_image, self.noisy_image) + 0.0001 * self.regularization_term(self.clean_image)

    def get_clean_image(self):
        return self.clean_image

original_dir = "C:/Users/V RAHUL/Codes/EE6310/Datasets/CBSD68/original_png/"
original_images_list = [os.path.join(original_dir, f) for f in os.listdir(original_dir)]
variance = 25
noisy_dir="C:/Users/V RAHUL/Codes/EE6310/Datasets/CBSD68/noisy"+str(variance)+"/"
noisy_images_list = [os.path.join(noisy_dir, f) for f in os.listdir(noisy_dir)]

fields=['Sl. No.','image name','noise variance','ssim value']
filename = "TVR_filter_results_"+str(variance)+".csv"
csvfile=open(filename,'w')
csvwriter=csv.writer(csvfile)
csvwriter.writerow(fields)

#print(noisy_images_list)
#print(original_images_list)
#print(len(noisy_images_list))

for i in range(len(noisy_images_list)):
    noisy_image_path=noisy_images_list[i]
    image_name=noisy_image_path.split("/")[-1]
    original_image_path= original_dir+"/"+image_name 
    original_image=cv2.imread(original_image_path)
    #plt.imshow(original_image)
    #plt.show()
    original_image=cv2.cvtColor(original_image,cv2.COLOR_BGR2GRAY)/255.0

    img=cv2.imread(noisy_image_path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)/255.0
    img=np.clip(img,0.0,1.0)

    # convert to torch tensor
    noisy_image: torch.tensor = kornia.image_to_tensor(img).squeeze()  # CxHxW

    tv_denoiser = TVDenoise(noisy_image)

    # define the optimizer to optimize the 1 parameter of tv_denoiser
    optimizer = torch.optim.SGD(tv_denoiser.parameters(), lr=0.1, momentum=0.9)

    # run the optimization loop
    num_iters = 500
    for j in range(num_iters):
        optimizer.zero_grad()
        loss = tv_denoiser()
        #print(loss)
        #if j % 25 == 0:
        #   print("Loss in iteration {} of {}:".format(i, num_iters), loss.item())
        loss.sum().backward()
        optimizer.step()

    # convert back to numpy
    img_clean: np.ndarray = kornia.tensor_to_image(tv_denoiser.get_clean_image())
    #plt.imshow(img_clean,cmap='gray')
    #plt.show()

    ssim_value=ssim(original_image,img_clean)
    print(ssim_value)

    rows=[i+1,image_name,variance,ssim_value]
    csvwriter.writerow(rows)

    output_dir='C:/Users/V RAHUL/Codes/EE6310/output_'+str(variance)+'/'+image_name
    cv2.imwrite(output_dir,img_clean*255.0)

    #print("[INFO] processed {} / {} images.".format(i+1,len(noisy_images_list)))

csvfile.close()
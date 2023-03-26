# Image denoising through weiner filter
image and video analysis course project

¬¬¬Implementation of classical image denoising methodologies:

Dataset used for comparative analysis:

	BSD68 : https://github.com/clausmichele/CBSD68-dataset 
	Set12. : dataset downloaded to google drive , https://figshare.com/articles/dataset/PSNR_results_of_denoising_by_different_methods_on_Set12_dataset_/21503325 

We can add AWGN of variance 25 and 50 to get noisy image and verify our algorithm 


	Kodak24: https://github.com/MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite
	Urban100 : downloaded to google drive
	SIDD. : will use for deep learning portion
	SenseNoise : will use for deep learning portion 

Comparative analysis metrics:


	Visual analysis
	Performance metric
	PSNR [Insert equation for PSNR]
	SSIM  [Insert equation for SSIM]



Gaussian noise Image denoising through Wiener filtering:

The Wiener filter is the MSE-optimal stationary linear filter for images degraded by additive noise and blurring. This filter is implemented under the assumption that the signal and the noise processes are second-order stationary. Hence , only noise processes with zero mean are considered.

The method is founded on considering images and noise as random variables and the objective is to find an estimate f ̂ of the uncorrupted image f such that mean square error between them is minimized. The error measure is defined as :
e^2=E{(f-f ̂ )^2 }

Where E{.} is the expected value. 

Below assumptions are made:

	Noise and image are uncorrelated
	One or the other has zero mean
	Intensity levels in the estimate are a linear function of the levels in the degraded image


Frequency domain expression for Wiener filter is given by:

F(u,v)=(H^* (u,v) S_f (u,v))/(S_f (u,v) |H(u,v)|^2+S_η (u,v) )
Where 

F(u,v)=Wiener filter transfer function
H(u,v)=degradation transfer function
H^* (u,v)=complex conjugate of H(u,v)
|H(u,v)|^2=H^* (u,v)H(u,v)
S_η (u,v)=power spectrum of the noise=variance in case of AWGN
S_f (u,v)=power spectrum of the ground truth image 


From the equations, it can be observed that Wiener filtering has two separate parts:
	An inverse filtering part
	A noise smoothing part

So it does high pass filtering through inverse filtering and removes the noise with a compression operation, which is a lowpass filtering operation. 



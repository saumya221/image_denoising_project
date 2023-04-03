import numpy as np

def MSE_PSNR(I,J):
    M,N = I.shape
    MSE = np.sum((I.astype("float") - J.astype("float")) ** 2)
    MSE /= M*N
    L = 255
    PSNR = 10*np.log10(L**2/MSE)
    
    return MSE,PSNR
def gaussian(sigma,size):
    gaussian_filter = np.zeros((size,size))

    for i in range(size):
        for j in range(size):
            gaussian_filter[i][j] = np.exp(-((i-size//2)**2+(j-size//2)**2)/(2*sigma**2))

    gaussian_filter/= np.sum(gaussian_filter)

    return gaussian_filter
def SSIM(I,J):
    
    ## Guassian Window for weight
    w = gaussian(0.5,11)
    M,N = I.shape
    
    C1,C2,C3  = (0.01*255)**2,(0.03*255)**2,((0.03*255)**2)/2
    L,C,S =  np.zeros((M,N)),np.zeros((M,N)),np.zeros((M,N))
    for i in range(5,M-5):
        for j in range(5,N-5):
            
            ## Luminance
            meanI = np.sum(w*I[i-5:i+6,j-5:j+6])
            meanJ = np.sum(w*J[i-5:i+6,j-5:j+6])
            L[i][j] = (2*meanI*meanJ + C1)/(meanI**2 + meanJ**2 + C1)

            
            ## Contrast
            sigI = np.sqrt(np.sum(w*(I[i-5:i+6,j-5:j+6] - meanI)**2))
            sigJ = np.sqrt(np.sum(w*(J[i-5:i+6,j-5:j+6] - meanJ)**2))
            C[i][j] = (2*sigI*sigJ + C2 )/(sigI**2 + sigJ**2 + C2)
            
            ## Similairity
            sigIJ = np.sum(w*(I[i-5:i+6,j-5:j+6] - meanI)*(J[i-5:i+6,j-5:j+6] - meanJ))
            S[i][j] = (sigIJ + C3)/(sigI*sigJ + C3)
            
    SSIM = L*C*S
    num = np.count_nonzero(SSIM)
    MSSIM  = np.sum(SSIM)/num
    SSIM_MAP = 255*SSIM
    return MSSIM,SSIM_MAP
            
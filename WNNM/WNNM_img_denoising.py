import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

def wnnm(img, patch_r, delta, c, K, sigma, threshold):
    search_window_r = 3*patch_r

    iter = 3

    pad = 4*patch_r
    img_pad = np.pad(img,pad)

    X_hat = img

    for n in range(K):
        X_hat = np.pad(X_hat,pad)
        y = X_hat + delta*(img_pad-X_hat)
        pixel_contributions = np.ones_like(img_pad)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                centre_patch = y[i+search_window_r:i+search_window_r+2*patch_r,j+search_window_r:j+search_window_r+2*patch_r]
                centre_patch_reshaped = np.reshape(centre_patch,(1,(2*patch_r)**2))
                dists = np.ones((2*search_window_r+1)**2)
                patches = np.zeros(((2*search_window_r+1)**2,(2*patch_r)**2))
                for k in range(2*search_window_r+1):
                    other_patch = y[i:i+2*pad,j+k:j+k+2*patch_r]
                    indices = np.reshape(np.arange((2*patch_r)**2),((1,(2*patch_r)**2)))+np.reshape((2*patch_r)*np.arange(other_patch.shape[0]-2*patch_r+1),(other_patch.shape[0]-2*patch_r+1,1))

                    other_patch = other_patch.flatten()
                    other_patch = np.reshape(other_patch[indices],(other_patch[indices].shape[0],(2*patch_r)**2))

                    dists[k*(2*search_window_r+1):(k+1)*(2*search_window_r+1)] = (np.sum(np.power(centre_patch_reshaped-other_patch,2),axis=1)/((2*patch_r)**2)).flatten()
                    patches[k*(2*search_window_r+1):(k+1)*(2*search_window_r+1),:] = other_patch
                

                idx = np.argsort(dists)
                Y = patches[idx[0:threshold],:].T
                U,S,V_T = linalg.svd(Y,full_matrices=False)

                singular_val_X_hat = np.sqrt(np.maximum(S**2-threshold*(sigma**2),0))


                for p in range(iter):
                    w = c*np.sqrt(threshold)/(singular_val_X_hat+1e-7)
                    singular_val_X_hat = np.maximum(singular_val_X_hat-w,0)

                Y_hat = U@np.diag(singular_val_X_hat)@V_T

                X_hat[i+search_window_r:i+search_window_r+2*patch_r,j+search_window_r:j+search_window_r+2*patch_r] = X_hat[i+search_window_r:i+search_window_r+2*patch_r,j+search_window_r:j+search_window_r+2*patch_r] + np.clip(Y_hat[:,0].reshape((2*patch_r,2*patch_r)),0,255)

                pixel_contributions[i+search_window_r:i+search_window_r+2*patch_r,j+search_window_r:j+search_window_r+2*patch_r] += 1

        X_hat = np.divide(X_hat[pad:-pad,pad:-pad],pixel_contributions[pad:-pad,pad:-pad]) 

    return X_hat
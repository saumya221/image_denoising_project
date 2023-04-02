close all
clc
clear all
original_files=dir('../../dataset/CBSD68-dataset/CBSD68/original_png/*');


for image_count=3:length(original_files)

    original_image_name=original_files(image_count).name;
    noisy_image_name='../../dataset/CBSD68-dataset/CBSD68/noisy15/'+string(original_image_name);
    original_image_path=fullfile('../../dataset/CBSD68-dataset/CBSD68/original_png',string(original_image_name));
    % read original input
    I=double(imread(original_image_path));

    %take only the first channel 
    I=I(:,:,1);

    %convert the shape to a shape divisible by 4
    [m,n]=size(I);
    m1=m-rem(m,4);
    n1=n-rem(n,4);
    I=imresize(I,[m1,n1]);
    
    %repeat the same process for noisy image 
    Inoise=double(imread(noisy_image_name));
    Inoise=Inoise(:,:,1);
    Inoise=imresize(Inoise,[m1,n1]);

    %select the 4 coefficient lowpass filter for wavelets
    lpfCoeff =[0.48296 0.83652 0.22414 -0.12941];
    % Change level of Decmposition HERE
    J = 4;
    
    % perform discrete wavelet transformation
    [C, S, wc] = discreteWavletTrans(Inoise, J, lpfCoeff);

    %estimation of noise level
    nEle = S(J,1) * S(J,2);
    hf = [C(1, nEle+1:2*nEle) C(1, 2*nEle+1:3*nEle) C(1, 3*nEle+1:4*nEle)];
    
    %calculate sigma
    sigma=median(abs(hf))/0.6745;
    
    threshold=3*sigma;
    
    % Soft thresholding
    CSoft = (sign(C).*(abs(C)-threshold)).*((abs(C)>threshold));
    
    %Hard Thresholding
    CHard = C.*((abs(C)>threshold));
    
    %reconstruction with soft and hard thresholds
    imageReconstSoft = InvdiscreteWavletTrans(CSoft, S, J, lpfCoeff);
    imageReconstHard = InvdiscreteWavletTrans(CHard, S, J, lpfCoeff);

    
    img=imageReconstHard;
    img=img-min(min(min(img)));
    img=img/max(max(max(img)));
    img=imresize(img,[m,n]);
    output_name='output_15/'+string(original_image_name);
    imwrite(img,output_name);
    string_to_print='processed '+string(image_count)

end 
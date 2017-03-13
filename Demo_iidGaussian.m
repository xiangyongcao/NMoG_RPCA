% Demo on i.i.d. Gaussian Noise
clear,clc
currentFolder = pwd;
addpath(genpath(currentFolder))
load('pure_DCmall.mat');
[M,N,B] = size(Ori_H);

muOn = 0;                     % muOn = 0: set mu as 0 without updating;
                              % muOn = 1: update mu in each iteration.
Rank = 5;                     % objective rank of low rank component
param.initial_rank = 30;      % initial rank of low rank component
param.rankDeRate = 7;         % the number of rank reduced in each iteration
param.mog_k = 1;              % the number of component reduced in each band      
param.lr_init = 'SVD';
param.maxiter = 30;         
param.tol = 1e-4;
param.display = 1; 
[prior, model] = InitialPara(param,muOn,B);        % set hyperparameters and initialize model parameters

for num = 1:20    
Noi_H = Ori_H + randn(M,N,B)*0.05;       % add noise
Y = reshape(Noi_H,M*N,B);
tic
[Model,Lr_model] = NMoG_RPCA(Y,Rank,param,model,prior);
time(num) = toc;
U = Lr_model.U;
V = Lr_model.V;
Denoi_HSI = reshape(U*V',size(Ori_H));
[PSNR(:,num),MPSNR(num),SSIM(:,num),MSSIM(num)] = zhibiao(Ori_H,Denoi_HSI); 
end

disp('*********************** DC_iidGaussian ************************'); 
meanPSNR = mean(MPSNR);
varPSNR = var(MPSNR);
meanSSIM = mean(MSSIM);
varSSIM = var(MSSIM);
Time = mean(time);
vTime = var(time);
disp(['MPSNR:',num2str(MPSNR),'   varMPSNR:',num2str(varPSNR),'   MSSIM:',num2str(MSSIM),...
    '   varMSSIM:',num2str(varSSIM),'   time:',num2str(Time),'   vartime:',num2str(vTime)]);


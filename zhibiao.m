function  [PSNR,MPSNR,SSIM,MSSIM] = zhibiao(Ori_H,Denoi_HSI)
[M,N,B] = size(Ori_H);
T1 = reshape(Ori_H*255,M*N,B);
T2 = reshape(Denoi_HSI*255,M*N,B);
temp = reshape(sum((T1 -T2).^2),B,1)/(M*N);
PSNR = 20*log10(255)-10*log10(temp);
MPSNR = mean(PSNR);
 for i=1:B
[mssim, ssim_map]=ssim_index(Ori_H(:,:,i)*255, Denoi_HSI(:,:,i)*255);
SSIM(i) = mssim;
 end
 MSSIM = mean(SSIM);

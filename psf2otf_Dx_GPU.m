function [otf] = psf2otf_Dx_GPU(outSize)

psf = single(zeros(outSize));
psf(1, 1) = -1;
psf(1, end) = 1;
psf = gpuArray(psf);
otf = fft2(psf);
function [otf] = psf2otf_Dy_GPU(outSize)

psf = single(zeros(outSize));
psf(1, 1) = -1;
psf(end, 1) = 1;
psf = gpuArray(psf);
otf = fftn(psf);
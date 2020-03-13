function [otf] = psf2otf_Dx(outSize)

psf = single(zeros(outSize));
psf(1, 1) = -1;
psf(1, end) = 1;
otf = fft2(psf);
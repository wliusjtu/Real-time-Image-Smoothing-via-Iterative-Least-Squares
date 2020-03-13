function [otf] = psf2otf_Dy(outSize)

psf = single(zeros(outSize));
psf(1, 1) = -1;
psf(end, 1) = 1;
otf = fft2(psf);
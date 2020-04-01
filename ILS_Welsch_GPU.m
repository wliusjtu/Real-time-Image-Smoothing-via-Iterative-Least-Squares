%   Distribution code Version 1.0 -- 02/31/2020 by Wei Liu Copyright 2020
%
%   The code is created based on the method described in the following paper 
%   [1] "Real-time Image Smoothing via Iterative Least Squares", Wei Liu, Pingping Zhang, 
%        Xiaolin Huang, Jie Yang, Chunhua Shen and Ian Reid, ACM Transactions on Graphics, 
%        presented at SIGGRAPH 2020.
%  
%   The code and the algorithm are for non-comercial use only.


%  ---------------------- Input------------------------
%  F:              input image, can be gray image or RGB color image
%  lambda:     \lambda in Eq.(1), control smoothing strength
%  gamma:       the \gamma in the Welsch's penalty in Eq. (18)
%  iter:           iteration number of the ILS 

%  ---------------------- Output------------------------
%  U:             smoothed image

function U =ILS_Welsch_GPU(F, lambda, gamma, iter)

F = gpuArray(single(F)); % 'single' precision is very important to reduce the computational cost

c = 2;

[N, M, D] = size(F);
sizeI2D = [N, M];

otfFx = psf2otf_Dx_GPU(sizeI2D); % equal to otfFx = psf2otf(fx, sizeI2D) where fx = [1, -1];
otfFy = psf2otf_Dy_GPU(sizeI2D); % equal to otfFy = psf2otf(fy, sizeI2D) where fy = [1; -1];

Denormin = abs(otfFx).^2 + abs(otfFy ).^2;
Denormin = repmat(Denormin, [1, 1, D]);
Denormin = 1 + 0.5 * c * lambda * Denormin;

U = F;  % smoothed image

Normin1 = fft2(U);

for k = 1: iter
    
    % Intermediate variables \mu update, in x-axis and y-axis direction
    u_h = [diff(U,1,2), U(:,1,:) - U(:,end,:)];
    u_v = [diff(U,1,1); U(1,:,:) - U(end,:,:)];
        
    mu_h = c .* u_h - 2 .* u_h .* exp(- u_h .* u_h / (2 * gamma^2));
    mu_v = c .* u_v - 2 .* u_v .* exp(- u_v .* u_v / (2 * gamma^2));
    
    % Update the smoothed image U
    Normin2_h = [mu_h(:,end,:) - mu_h(:, 1,:), - diff(mu_h,1,2)];
    Normin2_v = [mu_v(end,:,:) - mu_v(1, :,:); - diff(mu_v,1,1)];
    
    FU = (Normin1 + 0.5 * lambda * (fft2(Normin2_h + Normin2_v))) ./ Denormin;
    U = real(ifft2(FU));

    Normin1 = FU;  % This helps to further enlarge the smoothing strength
    
end

U = gather(U);

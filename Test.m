clear
close all

lambda = 1;
iter = 4;
p = 0.8;
eps = 0.0001; 

Img = im2double((imread('flower.png')));
Smoothed = ILS_LNorm(Img, lambda, p, eps, iter);
% Smoothed = ILS_LNorm_GPU(Img, lambda, p, eps, iter);

Diff = Img - Smoothed;
ImgE = Img + 3 * Diff;

figure; imshow(Smoothed)
figure; imshow(ImgE)

%%
Img = im2double(imread('clip_art.jpg'));
lambda = 30;
gamma = 10/255;
iter = 10;

Smoothed = ILS_Welsch(Img, lambda, gamma, iter);
% Smoothed = ILS_Welsch_GPU(Img, lambda, gamma, iter);

figure; imshow(Img)
figure; imshow(Smoothed)

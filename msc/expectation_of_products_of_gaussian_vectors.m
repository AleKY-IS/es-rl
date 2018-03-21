clear
clc
close all

x = sym('x', [5,1],'real');

x*x'*x

mean(randn(10000,1) .* randn(10000,1).^2)

mu = [1,2];
Sigma = [3, 0.5; 0.5, 2];
R = chol(Sigma);
x = mu + randn(1000000,2) * R;
mean(x(:,1).^3)
mean(x(:,1) .* x(:,2).^2)

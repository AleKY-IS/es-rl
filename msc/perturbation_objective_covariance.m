
clear
clc

n = 10000;

f = @(eps) mean(5 + eps)/mean(eps.^2);

eps1 = randn(n,1);
eps2 = randn(n,1);

cov(eps1,eps2)

cov(f(eps1)*eps1, f(eps2)*eps2)

cov(f(eps1)*eps1, f(-eps1)*(-eps1))




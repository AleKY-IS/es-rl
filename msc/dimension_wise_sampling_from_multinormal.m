clear
clc
close all

N = 10^5;
sigma = [3,3;3,5];
R=chol(sigma);


s = randn(N,2)*R;
s1 = s(:,1);

s = randn(N,2)*R;
s2 = s(:,2);

s = randn(N,2)*R;

figure;
subplot(1,2,1)
plot(s1,s2,'o')
subplot(1,2,2)
plot(s(:,1),s(:,2),'o')


clear

N = 100;


eps = randn(2, N);

Sigma = [2, 1.5; 1.5, 4];
L = chol(Sigma, 'lower');
mu = [2; 4];

s = mu + L * eps;
figure()
scatter(s(1,:), s(2,:))

gmu = inv(L)' * eps;
figure()
scatter(gmu(1,:), gmu(2,:))

gSigma = 



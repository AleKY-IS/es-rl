clear

s1 = sym('s1','real');
s2 = sym('s2','real');
s3 = sym('s3','real');
x = sym('x', [3,1],'real');
mu = sym('mu', [3,1],'real');
S = [s1, 0, 0;...
     0, s2, 0;...
     0, 0, s3];
s = [s1; s2; s3]; % diag of S


disp('(x - mu)'' * inv(S) * (x - mu) = ')
disp((x - mu)' * inv(S) * (x - mu))

disp('s^(-1)''(x-mu)^2 = ')
disp((1./s)' * (x-mu).^2)




disp('S*x*x''*S = ')
disp(S*x*x'*S)


s_example = [3;2;5];
log(prod(s_example))
sum(log(s_example))




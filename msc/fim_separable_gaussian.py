# coding: utf-8

import numpy as np


Sigma = np.diag([2,4,1])
dmu = np.array([0,1,0])
dmu.T @ Sigma @ dmu


Sigma = np.diag([2,4,1,3])
dSigma1 = np.zeros((4,4))
dSigma1[0,0] = 2

dSigma3 = np.zeros((4,4))
dSigma3[2,2] = 1
Sigmainv = np.linalg.inv(Sigma)

print('Sigma')
print(Sigma)
print('dSigma1')
print(dSigma1)
print('dSigma3')
print(dSigma3)
print('Sigmainv')
print(Sigmainv)

print('Sigmainv @ dSigma1 @ Sigmainv @ dSigma3 = ')
print(Sigmainv @ dSigma1 @ Sigmainv @ dSigma3)
print('Sigmainv @ dSigma3 @ Sigmainv @ dSigma3 = ')
print(Sigmainv @ dSigma3 @ Sigmainv @ dSigma3)
print('Sigmainv @ dSigma1 @ Sigmainv @ dSigma1 = ')
print(Sigmainv @ dSigma1 @ Sigmainv @ dSigma1)


d = 4
Sigma = np.diag(np.random.randint(1, high=9, size=d))
Sigmainv = np.linalg.inv(Sigma)
dSigmas = []
for i in range(d):
    dSigmas.append(np.zeros((d, d)))
    dSigmas[i][i, i] = Sigma[i, i].copy()

I_beta = np.zeros((d, d))
for i in range(d):
    for j in range(d):
        prod = Sigmainv @ dSigmas[i] @ Sigmainv @ dSigmas[j]
        tr = prod.trace()
        I_beta[i, j] = 0.5 * tr
        print('(' + str(i) + ', ' + str(j) + ')')
        print(prod, tr)
print(I_beta)
    
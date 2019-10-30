# QR factorization to solve Ax = b

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

def data_set_one(n, start, stop, eps):

    x = np.linspace(start, stop, n)
    r = np.random.rand(n)*eps
    y = x * (np.cos(r + 0.5*x**3) + np.sin(0.5*x**3))
    return x, y

def data_set_two(n, start, stop, eps):
    x = np.linspace(start, stop, n)
    r = np.random.rand(n)*eps
    y = 4*x**5 - 5*x**4 - 20*x**3 + 10*x**2 + 40*x + 10 + r
    return x, y

def designmatrix(x, order):
    mat = np.zeros((len(x),order))
    mat[:,0] = 1
    for i in range(1,order):
        mat[:,i] = x**i
    return mat

def QRsolve(X, y):

    Q, R = np.linalg.qr(X)
    ystar = Q.T @ y

    n = len(ystar)
    beta = np.zeros(n)
    for i in range(n-1, -1, -1):
        tmp = ystar[i]
        for j in range(n-1, i, -1):
            tmp -= beta[j]*R[i,j]
        beta[i] = tmp/R[i,i]
    ypredict = X @ beta
    return ypredict

def cholesky(A):

    A = np.array(A)
    n = A.shape[0]
    L = np.zeros((n,n))

    for row in range(n):
        for col in range(row+1):
            tmp_sum = np.dot(L[row,:col], L[col,:col])
            if (row == col):
                L[row, col] = np.sqrt(A[row,row] - tmp_sum)
            else:
                L[row,col] = (1.0 / L[col,col]) * (A[row,col] - tmp_sum)
    return L

def cholesky_solve(X, y):

    L = cholesky(X.T @ X)
    n = L.shape[0]

    y_1 = X.T @ y

    g = np.zeros(n)
    for i in range(0, n):
         s = y_1[i]
         for j in range(0,i):
              s = s - L[i,j]*g[j]
         g[i] = s / L[i,i]

    beta = np.zeros(n)
    for i in range(n-1, -1, -1):
        tmp = g[i]
        for j in range(n-1, i, -1):
            tmp -= beta[j]*L.T[i,j]
        beta[i] = tmp/L.T[i,i]

    fit = X @ beta
    return fit


if __name__ == '__main__':

    a, b = data_set_one(30, -2, 2, 1)
    X = designmatrix(a, 8)
    fit = cholesky_solve(X, b)
    fig = plt.figure(0)
    plt.scatter(a, b, label = 'data 1')
    plt.plot(a, fit, 'g', label = 'LS fit m=8')
    plt.legend()
    plt.show()
    fig.savefig('data1_m8')

    X_ = designmatrix(a, 3)
    fit_ = cholesky_solve(X_, b)
    fig1 = plt.figure(1)
    plt.scatter(a, b, label = 'data 1')
    plt.plot(a, fit_, 'g', label = 'LS fit m=3')
    plt.legend()
    plt.show()
    fig1.savefig('data1_m3')



    c, d = data_set_two(30, -2, 2, 1)
    X1 = designmatrix(c, 8)
    fit1 = cholesky_solve(X1, d)
    fig2 = plt.figure(2)
    plt.scatter(c,d, label = 'data 2')
    plt.plot(c, fit1, 'g', label = 'LS fit m=8')
    plt.legend()
    plt.show()
    fig2.savefig('data2_m8')

    X1_ = designmatrix(c, 3)
    fit1_ = cholesky_solve(X1_, d)
    fig3 = plt.figure(3)
    plt.scatter(c,d, label = 'data 2')
    plt.plot(c, fit1_, 'g', label = 'LS fit m=3')
    plt.legend()
    plt.show()
    fig3.savefig('data2_m3')

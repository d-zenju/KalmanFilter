#coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from LinearKalmanFilter import LinearKalmanFilter


class KalmanFilter:
    def __init__(self, A, B, u, Q, H, R, mu, Sigma):
        self.A = A
        self.B = B
        self.u = u
        self.Q = Q
        self.H = H
        self.R = R
        
        self.mu = mu
        self.Sigma = Sigma
        
    def update(self, y):
        mu_ = self.A * self.mu + self.B * self.u
        Sigma_ = self.Q + self.A * self.Sigma * self.A.T
        error = y - self.H * mu_
        S = self.H * Sigma_ * self.H.T + self.R
        K = Sigma_ * self.H.T * S.I
        self.mu = mu_ + K * error
        self.Sigma = Sigma_ - K * self.H * Sigma_
        
        return self.mu


def main():
    T = 30
    x = np.mat([[0], [0]])
    X = [np.mat([[0], [0]])]
    Y = [np.mat([[0], [0], [0]])]
    
    A = np.mat([[1, 0], [0, 1]])
    B = np.mat([[1, 0], [0, 1]])
    u = np.mat([[2], [2]])
    Q = np.mat([[1, 0], [0, 1]])
    
    H = np.mat([[1, 0], [0, 1]])
    R = np.mat([[2, 0], [0, 2]])
    
    for i in range(T):
        x = A * x + B * u + np.random.multivariate_normal([0, 0], Q, 1).T
        X.append(x)
        y = H * x + np.random.multivariate_normal([0, 0], R, 1).T
        Y.append(y)
    
    mu = np.mat([[0], [0]])
    Sigma = np.mat([[0, 0], [0, 0]])
    M = [mu]
    lkf = LinearKalmanFilter(mu, Sigma, A, B, u, H, Q, R)
    #lkf = KalmanFilter(A, B, u, Q, H, R, mu, Sigma)
    for i in range(T):
        #mu, Sigma = lkf.update(Y[i+1])
        mu = lkf.update(Y[i+1])
        M.append(mu)
    
    a, b = np.array(np.concatenate(X, axis = 1))
    plt.plot(a, b, 'rs-')
    a, b = np.array(np.concatenate(M, axis = 1))
    plt.plot(a, b, 'bo-')
    plt.show()


if __name__ == '__main__':
    main()
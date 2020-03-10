__author__ = 'Haohan Wang'

import numpy as np
from numpy import linalg

class Lasso:
    def __init__(self, lam=1., lr=1., tol=1e-5, logistic=False):
        self.lam = lam
        self.lr = lr
        self.tol = tol
        self.decay = 0.5
        self.maxIter = 500
        self.logistic = logistic

    def setLambda(self, lam):
        self.lam = lam

    def setLogisticFlag(self, logistic):
        self.logistic = logistic

    def setLearningRate(self, lr):
        self.lr = lr

    def setMaxIter(self, a):
        self.maxIter = a

    def setTol(self, t):
        self.tol = t

    def fit(self, X, y):
        X0 = np.ones(len(y)).reshape(len(y), 1)
        X = np.hstack([X, X0])
        shp = X.shape
        self.beta = np.zeros([shp[1], 1])
        resi_prev = np.inf
        resi = self.cost(X, y)
        step = 0
        while np.abs(resi_prev - resi) > self.tol and step < self.maxIter:
            keepRunning = True
            resi_prev = resi
            runningStep = 0
            while keepRunning and runningStep < 10:
                runningStep += 1
                prev_beta = self.beta
                pg = self.proximal_gradient(X, y)
                self.beta = self.proximal_proj(self.beta - pg * self.lr)
                keepRunning = self.stopCheck(prev_beta, self.beta, pg, X, y)
                if keepRunning:
                    self.lr = self.decay * self.lr
            step += 1
            resi = self.cost(X, y)
        return self.beta

    def cost(self, X, y):
        if self.logistic:
            tmp = (np.dot(X, self.beta)).T
            return -0.5 * np.sum(y*tmp - np.log(1+np.exp(tmp))) + self.lam * linalg.norm(
                self.beta, ord=1)
        else:
            return 0.5 * np.sum(np.square(y - np.dot(X, self.beta)).transpose()) + self.lam * linalg.norm(
                self.beta, ord=1)

    def proximal_gradient(self, X, y):
        if self.logistic:
            return -np.dot(X.transpose(), (y.reshape((y.shape[0], 1)) - 1. / (1 + np.exp(-np.dot(X, self.beta)))))
        else:
            return -np.dot(X.transpose(), (y.reshape((y.shape[0], 1)) - (np.dot(X, self.beta))))

    def proximal_proj(self, B):
        t = self.lam * self.lr
        zer = np.zeros_like(B)
        result = np.maximum(zer, B - t) - np.maximum(zer, -B - t)
        return result

    def predict(self, X):
        X0 = np.ones(X.shape[0]).reshape(X.shape[0], 1)
        X = np.hstack([X, X0])
        if not self.logistic:
            return np.dot(X, self.beta)
        else:
            t = 1. / (1 + np.exp(-np.dot(X, self.beta)))
            y = np.zeros_like(t)
            y[t>0.5] = 1
            return y

    def getBeta(self):
        self.beta = self.beta.reshape(self.beta.shape[0])
        return self.beta[:-1]

    def stopCheck(self, prev, new, pg, X, y):
        if np.square(linalg.norm((y - (np.dot(X, new))))) <= \
                                np.square(linalg.norm((y - (np.dot(X, prev))))) + np.dot(pg.transpose(), (
                            new - prev)) + 0.5 * self.lam * np.square(linalg.norm(prev - new)):
            return False
        else:
            return True
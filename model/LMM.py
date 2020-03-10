__author__ = 'Haohan Wang'

import scipy.optimize as opt

import sys
sys.path.append('../')

from helpingMethods import *

class LinearMixedModel:
    def __init__(self, numintervals=100, ldeltamin=-5, ldeltamax=5, mode='lmm', alpha=0.05, fdr=False, lowRankFit=False, tau=None):
        self.numintervals = numintervals
        self.ldeltamin = ldeltamin
        self.ldeltamax = ldeltamax
        self.mode = mode
        self.alpha = alpha
        self.fdr = fdr
        self.lowRankFit = lowRankFit
        self.tau = tau

    def correctData(self, X, K, Kva, Kve, y):
        [n_s, n_f] = X.shape
        assert X.shape[0] == y.shape[0], 'dimensions do not match'
        assert K.shape[0] == K.shape[1], 'dimensions do not match'
        assert K.shape[0] == X.shape[0], 'dimensions do not match'

        if self.tau is not None:
            K[K <= self.tau] = 0

        if y.ndim == 1:
            y = scipy.reshape(y, (n_s, 1))


        if self.lowRankFit:
            S, U, ldelta0 = self.train_nullmodel_lowRankFit(y, K, S=Kva, U=Kve, numintervals=self.numintervals,
                                                            ldeltamin=self.ldeltamin, ldeltamax=self.ldeltamax,
                                                            p=n_f)
        else:
            S, U, ldelta0 = self.train_nullmodel(y, K, S=Kva, U=Kve, numintervals=self.numintervals,
                                                 ldeltamin=self.ldeltamin, ldeltamax=self.ldeltamax, p=n_f)

        delta0 = scipy.exp(ldelta0)
        Sdi = 1. / (S + delta0)
        Sdi_sqrt = scipy.sqrt(Sdi)
        SUX = scipy.dot(U.T, X)
        # SUX = SUX * scipy.tile(Sdi_sqrt, (n_f, 1)).T
        for i in range(n_f):
            SUX[:, i] = SUX[:, i] * Sdi_sqrt.T
        SUy = scipy.dot(U.T, y)
        SUy = SUy * scipy.reshape(Sdi_sqrt, (n_s, 1))

        return SUX, SUy

    def fit(self, X, K, Kva, Kve, y):
        [n_s, n_f] = X.shape
        assert X.shape[0] == y.shape[0], 'dimensions do not match'
        assert K.shape[0] == K.shape[1], 'dimensions do not match'
        assert K.shape[0] == X.shape[0], 'dimensions do not match'

        if self.tau is not None:
            K[K<=self.tau] = 0

        if y.ndim == 1:
            y = scipy.reshape(y, (n_s, 1))

        X0 = np.ones(len(y)).reshape(len(y), 1)

        if self.lowRankFit:
            S, U, ldelta0 = self.train_nullmodel_lowRankFit(y, K, S=Kva, U=Kve, numintervals=self.numintervals,
                                                 ldeltamin=self.ldeltamin, ldeltamax=self.ldeltamax, p=n_f)
        else:
            S, U, ldelta0 = self.train_nullmodel(y, K, S=Kva, U=Kve, numintervals=self.numintervals,
                                                                 ldeltamin=self.ldeltamin, ldeltamax=self.ldeltamax, p=n_f)

        delta0 = scipy.exp(ldelta0)
        Sdi = 1. / (S + delta0)
        Sdi_sqrt = scipy.sqrt(Sdi)
        SUX = scipy.dot(U.T, X)
        for i in range(n_f):
            SUX[:, i] = SUX[:, i] * Sdi_sqrt.T
        SUy = scipy.dot(U.T, y)
        SUy = SUy * scipy.reshape(Sdi_sqrt, (n_s, 1))
        SUX0 = scipy.dot(U.T, X0)
        SUX0 = SUX0 * scipy.tile(Sdi_sqrt, (1, 1)).T

        self.pvalues = self.hypothesisTest(SUX, SUy, X, SUX0, X0)

        return np.exp(ldelta0)

    def rescale(self, a):
        return a / np.max(np.abs(a))

    def selectValues(self, Kva):
        r = np.zeros_like(Kva)
        n = r.shape[0]
        tmp = self.rescale(Kva)
        ind = 0
        for i in range(n / 2, n - 1):
            if tmp[i + 1] - tmp[i] > 1.0 / n:
                ind = i + 1
                break
        r[ind:] = Kva[ind:]
        r[n - 1] = Kva[n - 1]
        return r

    def fdrControl(self):
        tmp = self.pvalues
        tmp = sorted(tmp)
        threshold = 1e-8
        n = len(tmp)
        for i in range(n):
            if tmp[i] < (i+1)*self.alpha/n:
                threshold = tmp[i]
        self.pvalues[self.pvalues>threshold] = 1

    def getPvalues(self):
        if not self.fdr:
            # self.beta[self.beta < -np.log(self.alpha)] = 0
            return self.pvalues
        else:
            self.fdrControl()
            return self.pvalues

    def getEstimatedBeta(self):
        return self.estimatedBeta

    def hypothesisTest(self, UX, Uy, X, UX0, X0):
        [m, n] = X.shape
        p = []
        betas = []
        for i in range(n):
            if UX0 is not None:
                UXi = np.hstack([UX0, UX[:, i].reshape(m, 1)])
                XX = matrixMult(UXi.T, UXi)
                XX_i = linalg.pinv(XX)
                beta = matrixMult(matrixMult(XX_i, UXi.T), Uy)
                Uyr = Uy - matrixMult(UXi, beta)
                Q = np.dot(Uyr.T, Uyr)
                sigma = Q * 1.0 / m
            else:
                Xi = np.hstack([X0, UX[:, i].reshape(m, 1)])
                XX = matrixMult(Xi.T, Xi)
                XX_i = linalg.pinv(XX)
                beta = matrixMult(matrixMult(XX_i, Xi.T), Uy)
                Uyr = Uy - matrixMult(Xi, beta)
                Q = np.dot(Uyr.T, Uyr)
                sigma = Q * 1.0 / m
            betas.append(beta[1][0])
            ts, ps = tstat(beta[1], XX_i[1, 1], sigma, 1, m)
            if -1e30 < ts < 1e30:
                p.append(ps)
            else:
                p.append(1)
            # print beta[1][0], XX_i[1, 1], sigma, ps
        p = np.array(p)
        p[p<=1e-100] = 1e-100
        self.estimatedBeta = np.array(betas)
        return p

    def train_nullmodel(self, y, K, S=None, U=None, numintervals=500, ldeltamin=-5, ldeltamax=5, scale=0, mode='lmm', p=1):
        ldeltamin += scale
        ldeltamax += scale

        if S is None or U is None:
            S, U = linalg.eigh(K)

        Uy = scipy.dot(U.T, y)

        # grid search
        nllgrid = scipy.ones(numintervals + 1) * scipy.inf
        ldeltagrid = scipy.arange(numintervals + 1) / (numintervals * 1.0) * (ldeltamax - ldeltamin) + ldeltamin
        for i in scipy.arange(numintervals + 1):
            nllgrid[i] = nLLeval(ldeltagrid[i], Uy, S) # the method is in helpingMethods

        nllmin = nllgrid.min()
        ldeltaopt_glob = ldeltagrid[nllgrid.argmin()]

        for i in scipy.arange(numintervals - 1) + 1:
            if (nllgrid[i] < nllgrid[i - 1] and nllgrid[i] < nllgrid[i + 1]):
                ldeltaopt, nllopt, iter, funcalls = opt.brent(nLLeval, (Uy, S),
                                                              (ldeltagrid[i - 1], ldeltagrid[i], ldeltagrid[i + 1]),
                                                              full_output=True)
                if nllopt < nllmin:
                    nllmin = nllopt
                    ldeltaopt_glob = ldeltaopt

        return S, U, ldeltaopt_glob

    def train_nullmodel_lowRankFit(self, y, K, S=None, U=None, numintervals=500, ldeltamin=-5, ldeltamax=5, scale=0, mode='lmm', p=1):
        ldeltamin += scale
        ldeltamax += scale

        if S is None or U is None:
            S, U = linalg.eigh(K)

        Uy = scipy.dot(U.T, y)

        S = self.selectValues(S)
        nllgrid = scipy.ones(numintervals + 1) * scipy.inf
        ldeltagrid = scipy.arange(numintervals + 1) / (numintervals * 1.0) * (ldeltamax - ldeltamin) + ldeltamin
        for i in scipy.arange(numintervals + 1):
            nllgrid[i] = nLLeval(ldeltagrid[i], Uy, S)  # the method is in helpingMethods

        nllmin = nllgrid.min()
        ldeltaopt_glob = ldeltagrid[nllgrid.argmin()]

        for i in scipy.arange(numintervals - 1) + 1:
            if (nllgrid[i] < nllgrid[i - 1] and nllgrid[i] < nllgrid[i + 1]):
                ldeltaopt, nllopt, iter, funcalls = opt.brent(nLLeval, (Uy, S),
                                                              (ldeltagrid[i - 1], ldeltagrid[i], ldeltagrid[i + 1]),
                                                              full_output=True)
                if nllopt < nllmin:
                    nllmin = nllopt
                    ldeltaopt_glob = ldeltaopt

        return S, U, ldeltaopt_glob
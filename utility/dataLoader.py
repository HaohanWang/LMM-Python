__author__ = 'Haohan Wang'

import sys
import numpy as np
import pysnptools
import operator


class FileReader():
    def __init__(self, fileName, imputation=True, fileType=None):
        self.fileName = fileName
        self.imputationFlag = imputation
        if fileType is None:
            self.fileType = 'plink'
        else:
            self.fileType = fileType

    def famReader(self, fileName):
        d = []
        text = [line.strip() for line in open(fileName)]
        for line in text:
            d.append(float(line.split()[-1]))
        return np.array(d)

    def imputation(self, X):
        print 'Missing genotype imputation ... '
        print 'This may take a while, use -m to skip this step'
        [n, p] = X.shape
        dis = np.zeros([n, n])
        for i in range(n):
            for j in range(i+1, n):
                d = np.nanmean(np.square(X[i,:]-X[j,:]))
                dis[i,j] = d
                dis[j,i] = d

        mx, xy = np.where(np.isnan(X)==1)

        missing = {}
        for x in mx:
            if x not in missing:
                missing[x] = 1
            else:
                missing[x] += 1

        ms = sorted(missing.items(), key=operator.itemgetter(1))
        ms.reverse()
        for (x, k) in ms:
            neighbors = np.argsort(dis[x,:])[1:]
            for i in range(n-1):
                n = neighbors[i]
                ind = np.where(np.isnan(X[x,:])==1)[0]
                if len(ind) == 0:
                    break
                X[x,ind] = X[n,ind]
        return X


    def simpleImputation(self, X):
        X[np.isnan(X)] = 0
        return X


    def readFiles(self):
        print 'Reading Data ...'
        X = None
        y = None
        Xname = None
        if self.fileType == 'plink':
            from pysnptools.snpreader import Bed
            snpreader = Bed(self.fileName+'.bed')
            snpdata = snpreader.read()
            X = snpdata.val
            Xname = snpdata.sid
            y = self.famReader(self.fileName+".fam")

        if self.fileType == 'csv':
            X = np.loadtxt(self.fileName+'.geno.csv', delimiter=',')
            y = np.loadtxt(self.fileName+'.pheno.csv', delimiter=',')
            try:
                Xname = np.loadtxt(self.fileName+'.marker.csv', delimiter=',')
            except:
                Xname = ['geno ' + str(i+1) for i in range(X.shape[1])]
        if self.imputationFlag:
            X = self.imputation(X)
            keep = True - np.isnan(y)
            return X[keep,:], y[keep], Xname
        else:
            X = self.simpleImputation(X)
            keep = np.isnan(y)==False
            return X[keep,:], y[keep], Xname

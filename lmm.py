__author__ = 'Haohan Wang'


# Main file for usage of (CS-LMM) Constrained Sparse multi-locus Linear Mixed Model
# Cite information:
# Wang H, Aragam B, Lee S, Xing EP, and Wu W.
# Discovering Weaker Genetic Associations Guided by Known Associations, with Application to Alcoholism and Alzheimer's Disease Studies
#

def printOutHead(): out.write("\t".join(["RANK", "SNP_ID", "EFFECT_SIZE_ABS"]) + "\n")


def outputResult(rank, id, beta):
    out.write("\t".join([str(x) for x in [rank, id, beta]]) + "\n")


from optparse import OptionParser, OptionGroup

usage = """usage: %prog [options] -n fileName
This program provides the basic usage to CS-LMM, e.g:
python cslmm.py -n data/mice.plink
	    """
parser = OptionParser(usage=usage)

dataGroup = OptionGroup(parser, "Data Options")
modelGroup = OptionGroup(parser, "Model Options")

## data options
dataGroup.add_option("-f", dest='fileType', default='plink', help="choices of input file type")
dataGroup.add_option("-n", dest='fileName', help="name of the input file")

## model options
modelGroup.add_option('-s', action='store_true', dest='select', default=False,
                      help='Construct Kinship with Selected Covariates (LMM-select)')
modelGroup.add_option('-l', action='store_true', dest='lowRank', default=False,
                      help='Construct Kinship with low Rank Matrix (truncated rank LMM)')
modelGroup.add_option('-t', dest='threshold', default=0,
                      help='Construct Kinship with thresholded kinship matrix')
modelGroup.add_option('-q', action='store_true', dest='quiet', default=False, help='Run in quiet mode')
modelGroup.add_option('-p', action='store_true', dest='plot', default=False, help='Generate Manhattan plot')
modelGroup.add_option('-m', action='store_true', dest='missing', default=False,
                      help='Run without missing genotype imputation')

## advanced options
parser.add_option_group(dataGroup)
parser.add_option_group(modelGroup)

(options, args) = parser.parse_args()


import sys
from utility.dataLoader import FileReader
from model.LMM import LinearMixedModel
from model.Lasso import Lasso
from model.helpingMethods import *

fileType = 0
IN = None

if len(args) != 0:
    parser.print_help()
    sys.exit()

outFile = options.fileName + '.output'

print ('Running ... ')

reader = FileReader(fileName=options.fileName, fileType=options.fileType, imputation=(not options.missing))
X, Y, Xname = reader.readFiles()

model = LinearMixedModel()

print ('Computation starts ... ')

if options.select:
    linearRegression = Lasso(lam=0)
    linearRegression.fit(X, Y)
    beta_lr = np.abs(linearRegression.getBeta())

    Xselected = selectTopX(beta_lr, X, X.shape[1], [0.01, ])[0]
    lmm = LinearMixedModel()
    K = matrixMult(Xselected, Xselected.T)/float(Xselected.shape[1])
    lmm.fit(X=X, K=K, Kva=None, Kve=None, y=Y)
    pvalue= lmm.getPvalues()
else:
    if options.threshold != 0:
        lmm = LinearMixedModel(tau=0.001)
        K = matrixMult(X, X.T) / float(X.shape[1])
        X, y = lmm.correctData(X=X, K=K, Kva=None, Kve=None, y=Y)
        K = matrixMult(X, X.T) / float(X.shape[1])
        lmm.fit(X=X, K=K, Kva=None, Kve=None, y=y)
        pvalue = lmm.getPvalues()
    elif options.lowRank:
        lmm = LinearMixedModel(lowRankFit=True)
        K = matrixMult(X, X.T) / float(X.shape[1])
        X, y = lmm.correctData(X=X, K=K, Kva=None, Kve=None, y=Y)
        K = matrixMult(X, X.T) / float(X.shape[1])
        lmm.fit(X=X, K=K, Kva=None, Kve=None, y=y)
        pvalue = lmm.getPvalues()
    else:
        lmm = LinearMixedModel()
        K = matrixMult(X, X.T) / float(X.shape[1])
        lmm.fit(X=X, K=K, Kva=None, Kve=None, y=Y)
        pvalue = lmm.getPvalues()



ind = np.where(pvalue != 0)[0]
bs = pvalue[ind].tolist()
xname = []
for i in ind:
    xname.append(i)

beta_name = zip(pvalue, Xname)
bn = sorted(beta_name)
bn.reverse()

out = open(outFile, 'w')
printOutHead()

for i in range(len(bn)):
    outputResult(i + 1, bn[i][1], bn[i][0])

out.close()

print ('\nComputation ends normally, check the output file at ', outFile)

if options.plot:
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=[100, 5])
    plt.scatter(range(pvalue.shape[0]), -np.log(pvalue))
    plt.savefig(outFile[:-7]+'.png')
    print ('\nManhattan plot drawn, check the output file at ', outFile)

![lmm-python](lmm.png "LMM-Python")

# LMM-Python 

Implementation of the Python Package of Linear Mixed Model, associated with the following paper:

Wang, H., Aragam, B., & Xing, E. P. Tradeoffs of Linear Mixed Models in Genome-wide Association Studies

## Introduction

LMM-Python is a python package of linear mixed model, including several popular methods used to calculate the kinship matrix, including

* with selected SNPs (LMM-select): 
    * [FaST-LMM-Select for addressing confounding from spatial structure and rare variants](https://www.ncbi.nlm.nih.gov/pubmed/23619783) 
* with low rank structure kinship: 
    * [Variable Selection in Heterogeneous Datasets: A Truncated-rank Sparse Linear Mixed Model with Applications to Genome-wide Association Studies](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5889139/)
* with masked kinship: 
    * [Two-Variance-Component Model Improves Genetic Prediction in Family Datasets](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4667134/)

## File Structure:

* [models/](https://github.com/HaohanWang/LMM-Python/tree/master/model) main method for the package
* [utility/](https://github.com/HaohanWang/LMM-Python/tree/master/utility) other helper files
* [lmm.py](https://github.com/HaohanWang/LMM-Python/blob/master/lmm.py) main entry point of using the package

## An Example Command:

```
python lmm.py -n data/mice.plink
```
#### Instructions
```
  Options:
  -h, --help          show this help message and exit

  Data Options:
    -f FILETYPE       choices of input file type
    -n FILENAME       name of the input file

  Model Options:
    -s                Construct kinship matrix with selected SNPs
    -l                Construct kinship matrix with low rank structure
    -t THRESHOLD      Construct kinship matrix with smaller values masked (smaller than the specificed THRESHOLD)
    -q                Run in quiet mode
    -m                Run without missing genotype imputation
    -p                Generate a simple Manhattan plot after running
```
#### Data Support
* The package currently supports CSV and binary PLINK files.
* Extensions to other data format can be easily implemented through `FileReader` in `utility/dataLoadear`. Feel free to contact us for the support of other data format.

## Python Users
Proficient python users can directly call the method with python code, see example starting at [Line 75](https://github.com/HaohanWang/LMM-Python/blob/master/lmm.py#L75)

## Installation (Not Required)
* Dependencies: 
    * numpy
    * scipy
    * pysnptool
    * matplotlib

You can install LMM-Python using pip by doing the following

```
   pip install git+https://github.com/HaohanWang/LMM-Python
```

You can also clone the repository and do a manual install.
```
   git clone https://github.com/HaohanWang/LMM-Python
   python setup.py install
```

## Contact
[Haohan Wang](http://www.cs.cmu.edu/~haohanw/)
&middot;
[@HaohanWang](https://twitter.com/HaohanWang)

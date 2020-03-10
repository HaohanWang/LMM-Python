from distutils.core import setup

setup(
      name='lmm-python',
      version='0.99',
      author = "Haohan Wang",
      author_email='haohanw@cs.cmu.edu',
      url = "https://github.com/HaohanWang/LMM-Python",
      description = "Tradeoffs of Linear Mixed Models in Genome-wide Association Studies",
      packages=['models', 'utility'],
      scripts=['lmm.py'],
    )
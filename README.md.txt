# This is the code for the paper 'ASkewSGD : An Annealed interval-constrained Optimisation method to train Quantized Neural Networks'


### This code evaluates the method ASkewSGD on 3 tasks:
- A toy convex problem (see askewsgd_comparison_logistic.ipynb)
- A toy non-convex pronlem (see askewsgd_comparison_2moons.ipynb)
- ImageNet dataset (see folder askewsgd_Imagenet and run 'python main_askewsgd.py')


#### The first experiments are available with jupyter Notebooks. The latter is mainly based on the implementation
from LUQ (https://openreview.net/forum?id=clwYez4n8e8), and adapted to our annealing approach ASkewSGD.
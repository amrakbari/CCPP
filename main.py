import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from regression_models.DTR import dtr
from regression_models.RFR import rfr
from regression_models.SVR import svr
from regression_models.multiple_linear_regression import multiple_linear_regression
from regression_models.polynomial_regression import polynomial_regression

dataset = pd.read_excel('Folds5x2_pp.xlsx')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
multiple_linear_regression(x, y)
print('________________________________________')
polynomial_regression(x, y)
print('________________________________________')
svr(x, y)
print('________________________________________')
dtr(x, y)
print('________________________________________')
rfr(x, y)
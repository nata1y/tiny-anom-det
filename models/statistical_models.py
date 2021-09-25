# from https://github.com/RuoyunCarina-D/Anomly-Detection-SARIMA/blob/master/SARIMA.py

import itertools
import numpy as np
from sklearn.metrics import mean_squared_log_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pandas as pd
import matplotlib.pyplot as plt


def fit_sarima(data):
    ############################################# Seasaonl decomposing ###############################################
    # res = seasonal_decompose(data.interpolate(), model='additive')
    # res.plot()

    ############################################# Model Selection######################################################

    # define the p, d and q parameters to take any value between 0 and 2
    p = d = q = range(0, 2)

    # generate all different combinations of p, d and q triplets
    pdq = list(itertools.product(p, d, q))

    # generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    best_aic = np.inf
    best_pdq = None
    best_seasonal_pdq = None
    res = None

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                tmp_mdl = SARIMAX(data,
                                  order=param,
                                  seasonal_order=param_seasonal,
                                  enforce_stationarity=True,
                                  enforce_invertibility=False)

                res = tmp_mdl.fit()
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_pdq = param
                    best_seasonal_pdq = param_seasonal
                    best_mdl = tmp_mdl

            except Exception as e:
                print(e)

    print("Best SARIMAX{}x{}12 model - AIC:{}".format(best_pdq, best_seasonal_pdq, best_aic))
    return best_mdl.fit()

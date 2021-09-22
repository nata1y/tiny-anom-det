import itertools
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX


def fit_sarima(data):
    # define the p, d and q parameters to take any value between 0 and 2
    p = d = q = range(0, 2)

    # generate all different combinations of p, d and q triplets
    pdq = list(itertools.product(p, d, q))

    seasonal = 52
    # generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], seasonal) for x in list(itertools.product(p, d, q))]

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
    return res

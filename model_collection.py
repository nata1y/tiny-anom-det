from skmultiflow.drift_detection import DDM, HDDM_W, HDDM_A, EDDM
from skopt.space import Categorical, Integer, Real
from skmultiflow.drift_detection.adwin import ADWIN

from drift_detectors.ecdd import ECDD
from models.nets import LSTM_autoencoder
from models.pso_elm.pso_elm_anomaly import PSO_ELM_anomaly
from models.sr.spectral_residual import THRESHOLD, MAG_WINDOW, SCORE_WINDOW, SpectralResidual
from models.statistical_models import SARIMA

models = {
          'pso-elm': (PSO_ELM_anomaly, [Integer(low=50, high=1000, name="n"),
                                        Real(low=0.0001, high=5.0, name="magnitude"),
                                        Real(low=0.0001, high=10.0, name="error_threshold")],
                      [300, 5.0, 5.0]),
          'sr': (SpectralResidual, [Real(low=0.01, high=1.0001, name='THRESHOLD'),
                                    Integer(low=1, high=30, name='MAG_WINDOW'),
                                    Integer(low=5, high=1000, name='SCORE_WINDOW'),
                                    Integer(low=1, high=100, name='sensitivity')],
                 [THRESHOLD, MAG_WINDOW, SCORE_WINDOW, 99]),
          'lstm': (LSTM_autoencoder, [Real(low=0.5, high=10.0, name='threshold')], [5.0]),
          'sarima': (SARIMA, [Real(low=0.5, high=5.0, name="conf_top"), Real(low=0.5, high=5.0, name="conf_botton")],
                    [1.2, 1.2])
          }


drift_detectors = {
    'adwin': (ADWIN, True),
    'ddm': (DDM, True),
    'eddm': (EDDM, True),
    'hddma': (HDDM_A, True),
    'hddmw': (HDDM_W, True),
    'ecdd': (ECDD, True),
    'no_da': (None, False)
}

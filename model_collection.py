from alibi_detect.cd import MMDDrift
from skmultiflow.drift_detection import DDM, PageHinkley, KSWIN, HDDM_W, HDDM_A, EDDM
from skopt.space import Categorical, Integer, Real
from skmultiflow.drift_detection.adwin import ADWIN
from models.nets import LSTM_autoencoder
from models.pso_elem.pso_elm_anomaly import PSO_ELM_anomaly
from models.sr.spectral_residual import THRESHOLD, MAG_WINDOW, SCORE_WINDOW, SpectralResidual
from models.statistical_models import SARIMA

#TODO: normalize function args order/collection + pass threshold type for plotting

models = {
          'pso-elm': (PSO_ELM_anomaly, [Integer(low=50, high=1000, name="n"),
                                        Real(low=0.0001, high=5.0, name="magnitude"),
                                        Real(low=0.0001, high=10.0, name="error_threshold")],
                      [300, 5.0, 5.0]),
          # 'sr': (SpectralResidual, [Real(low=0.01, high=1.0001, name='THRESHOLD'),
          #                           Integer(low=1, high=30, name='MAG_WINDOW'),
          #                           Integer(low=5, high=1000, name='SCORE_WINDOW'),
          #                           Integer(low=1, high=100, name='sensitivity')],
          #        [THRESHOLD, MAG_WINDOW, SCORE_WINDOW, 99]),
          # 'lstm': (LSTM_autoencoder, [Real(low=0.5, high=3.0, name='threshold')], [1.5]),
          # 'sarima': (SARIMA, [Real(low=0.5, high=5.0, name="conf_top"), Real(low=0.5, high=5.0, name="conf_botton")],
          #           [1.2, 1.2])
          }


drift_detectors = {
    'mmd': (MMDDrift, [Integer(low=100, high=5000, name="data_in_memory"),
                       Real(low=0.01, high=0.5, name="p_val")], [100, .05]),
    'adwin': (ADWIN, [Real(low=0.00001, high=0.9, name="delta")], [0.002]),
    'ddm': (DDM, [Real(low=1.0, high=20.0, name="out_control_level")], [3.0]),
    'eddm': (EDDM, [], []),
    'hddma': (HDDM_A, [Real(low=0.00001, high=0.9, name="drift_conf")], [0.001]),
    'hddmw': (HDDM_W, [Real(low=0.00001, high=0.9, name="drift_conf"), Real(low=0.00001, high=1.0, name="lambda")],
              [0.001, 0.05]),
    'kswin': (KSWIN, [Real(low=0.0001, high=0.01, name="alpha"),
                      Integer(low=100, high=1000, name='window_size')], [0.005, 1000]),
    'ph': (PageHinkley, [Real(low=0.00001, high=0.9, name="delta"), Integer(low=1, high=1000, name='threshold')],
           [0.005, 50])
}

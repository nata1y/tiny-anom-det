from alibi_detect.cd import MMDDrift, LSDDDrift
from sklearn.cluster import DBSCAN
from skmultiflow.drift_detection import DDM, PageHinkley, KSWIN, HDDM_W, HDDM_A, EDDM
from skopt.space import Categorical, Integer, Real
from skmultiflow.drift_detection.adwin import ADWIN
from tslearn.metrics import dtw

from models.driftpoint import DriftPointModel
from models.naive import NaiveDetector
from models.nets import LSTM_autoencoder
from models.sr.spectral_residual import THRESHOLD, MAG_WINDOW, SCORE_WINDOW, SpectralResidual
from models.statistical_models import SARIMA, ExpSmoothing

models = {
          # 'mogaal': (MOGAAL, [], []),
          # 'dbscan': (DBSCAN,
          #            [Integer(low=1, high=100, name='eps'), Integer(low=1, high=100, name='min_samples'),
          #             Categorical(['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan',
          #                           'nan_euclidean', dtw], name='metric')],
          #             [1, 2, dtw]),
          # 'lof': (LocalOutlierFactor, [Integer(low=1, high=1000, name='n_neighbors'),
          #                              Real(low=0.001, high=0.5, name="contamination")], [5, 0.1]),
          # scale gamma='scale'
          # 'ocsvm': (OneClassSVM,
          #           [Real(low=0.001, high=0.999, name='nu'), Categorical(['linear', 'rbf', 'poly'], name='kernel')],
          #           [0.85, 'poly']),
          # no norm
          # 'isolation_forest': (IsolationForest, [Integer(low=1, high=1000, name='n_estimators')], [100]),
          # 'isolation_forest': (IsolationForest, [Real(low=0.01, high=0.99, name='fraction')], [0.1]),
          # 'sr_alibi': (SR, [Real(low=0.01, high=10.0, name='threshold'),
          #                   Integer(low=1, high=anomaly_window, name='window_amp'),
          #                   Integer(low=1, high=anomaly_window, name='window_local'),
          #                   Integer(low=1, high=anomaly_window, name='n_est_points'),
          #                   Integer(low=1, high=anomaly_window, name='n_grad_points'),
          #                   Real(low=0.5, high=0.999, name='percent_anom')],
          #              [1.0, 20, 20, 10, 5, 0.95]),
          # 'seq2seq': (OutlierSeq2Seq, [Integer(low=1, high=100, name='latent_dim'),
          #                              Real(low=0.5, high=0.999, name='percent_anom')], [2, 0.95]),
          # 'sr': (SpectralResidual, [Real(low=0.01, high=1.0001, name='THRESHOLD'),
          #                           Integer(low=1, high=30, name='MAG_WINDOW'),
          #                           Integer(low=5, high=1000, name='SCORE_WINDOW'),
          #                           Integer(low=1, high=100, name='sensitivity')],
          #        [THRESHOLD, MAG_WINDOW, SCORE_WINDOW, 99]),
          'dp': (DriftPointModel, [], []),
          # 'lstm': (LSTM_autoencoder, [Real(low=0.0, high=20.0, name='threshold')], [1.5]),
          # 'seasonal_decomp': (DecomposeResidual, [], []),
          # 'sarima': (SARIMA, [Real(low=0.5, high=5.0, name="conf_top"), Real(low=0.5, high=5.0, name="conf_botton")],
          #            [1.2, 1.2]),
          # 'naive': (NaiveDetector, [Integer(low=2, high=100, name='k'), Categorical([0.0, 1.0], name='u'),
          #                           Real(low=0.01, high=3.0, name='c'), Real(low=0.01, high=100.0, name='b'),
          #                           Categorical([True, False], name='useabs')],
          #             [30, 1.0, 1.0, 1.0, False]),
          # 'es': (ExpSmoothing, [Integer(low=10, high=1000, name='sims')], [100]),
          # 'mmd-online': (MMDDriftOnlineTF, [], []),
          # 'lsdd': (LSDDDrift, [Real(low=0.000001, high=0.5, name="p_val")], [0.05])

          # 'ensemble': (Ensemble, [], []),
          # 'vae': (OutlierVAE, [Real(low=0.01, high=0.99, name='threshold'),
          #                      Integer(low=2, high=anomaly_window, name='latent_dim'),
          #                      Integer(low=1, high=100, name='samples'),
          #                      Real(low=0.5, high=0.999, name='percent_anom')],
          #         [0.9, 100, 10, 0.95])
          }


drift_detectors = {
    # 'mmd': (MMDDrift, [Integer(low=100, high=5000, name="data_in_memory"),
    #                    Real(low=0.01, high=0.5, name="p_val")], [100, .05]),
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

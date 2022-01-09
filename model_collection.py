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
          'sr': (SpectralResidual, [Real(low=0.01, high=1.0001, name='THRESHOLD'),
                                    Integer(low=1, high=30, name='MAG_WINDOW'),
                                    Integer(low=5, high=1000, name='SCORE_WINDOW'),
                                    Integer(low=1, high=100, name='sensitivity')],
                 [THRESHOLD, MAG_WINDOW, SCORE_WINDOW, 99]),
          'lstm': (LSTM_autoencoder, [Real(low=0.5, high=3.0, name='threshold')], [1.5]),
          # 'seasonal_decomp': (DecomposeResidual, [], []),
          'sarima': (SARIMA, [Real(low=0.5, high=5.0, name="conf_top"), Real(low=0.5, high=5.0, name="conf_botton")],
                     [1.2, 1.2]),
          # 'lstm': (LSTM_autoencoder, [Real(low=0.5, high=3.0, name='threshold')], [1.5]),
          # 'dp': (DriftPointModel, [Real(low=0.5, high=5.0, name="conf_top"), Real(low=0.5, high=5.0, name="conf_botton"),
          #                          Real(low=0.00001, high=0.9, name="delta")], [1.2, 1.2, 0.002]),
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


anomaly_taxonomy = {
    # real_37, real_43, real_53, ec2_cpu_utilization_ac20cd, ec2_network_in_5abac7, speed_7578, 'Twitter_volume_KO'
    # Twitter_volume_UPS synthetic_2, synthetic_16 - ?
    '1a': ['real_1', 'real_2', 'real_3', 'real_4', 'real_6', 'real_10', 'real_11', 'real_12', 'real_13', 'real_15',
           'real_9', 'real_16', 'real_20', 'real_21', 'real_24', 'real_26', 'real_29', 'real_30', 'real_33', 'real_34',
           'real_38', 'real_39', 'real_41', 'real_45', 'real_50', 'real_51', 'real_52', 'real_60', 'real_61', 'real_62',
           'ambient_temperature_system_failure', 'ec2_cpu_utilization_5f5533', 'ec2_cpu_utilization_fe7f93',
           'ec2_disk_write_bytes_c0d644', 'ec2_request_latency_system_failure', 'elb_request_count_8c0756',
           'exchange-3_cpm', 'exchange-4_cpm', 'occupancy_6005', 'occupancy_t4013', 'speed_t4013', 'Twitter_volume_AAPL',
           'Twitter_volume_FB', 'synthetic_5', 'synthetic_7', 'synthetic_9', 'synthetic_10', 'synthetic_11',
           'synthetic_13', 'synthetic_14', 'synthetic_19', 'synthetic_21', 'synthetic_23', 'synthetic_25',
           'synthetic_27', 'synthetic_28', 'synthetic_30', 'synthetic_31', 'synthetic_35', 'synthetic_37',
           'synthetic_39', 'synthetic_41', 'synthetic_42', 'synthetic_43', 'synthetic_44', 'synthetic_47',
           'synthetic_48', 'synthetic_49', 'synthetic_51', 'synthetic_53', 'synthetic_55', 'synthetic_56',
           'synthetic_57', 'synthetic_58', 'synthetic_59', 'synthetic_60', 'synthetic_61', 'synthetic_62',
           'synthetic_63', 'synthetic_65', 'synthetic_67', 'synthetic_68', 'synthetic_69', 'synthetic_70',
           'synthetic_72', 'synthetic_73', 'synthetic_74', 'synthetic_75', 'synthetic_76', 'synthetic_77',
           'synthetic_79', 'synthetic_81', 'synthetic_83', 'synthetic_84', 'synthetic_85', 'synthetic_88',
           'synthetic_89', 'synthetic_90', 'synthetic_91', 'synthetic_93', 'synthetic_95', 'synthetic_97',
           'synthetic_98', 'synthetic_99', 'synthetic_100'],
    '1b': [],
    '4a': ['ec2_cpu_utilization_24ae8d', 'ec2_disk_write_bytes_1ef3de', 'nyc_taxi', 'synthetic_1', '',
           'synthetic_3', 'synthetic_4', 'synthetic_6', 'synthetic_8', 'synthetic_22', 'synthetic_15', 'synthetic_12',
           'synthetic_36', 'synthetic_40', 'synthetic_45', 'synthetic_50', 'synthetic_64', 'synthetic_66',
           'synthetic_78', 'synthetic_92', 'synthetic_96'],
    '7a': ['real_25', 'real_40', 'real_46', 'art_daily_flatmiddel', 'art_daily_jumpsdown', 'art_daily_jumpsup',
           'art_daily_nojumps'],
    '7b': ['real_8', 'ec2_cpu_utilization_53ea38'],
    '7c': ['ec2_cpu_utilization_ac20cd', 'grok_asg_anomaly', 'rds_cpu_utilization_cc0c53'],
    '7d': ['real_7', 'real_17', 'real_19', 'real_22', 'real_24', 'real_28', 'real_31', 'real_32', 'real_42', 'real_58',
           'real_63', 'real_66', 'real_67', 'cpu_utilization_asg_misconfiguration'],
    '7e': ['real_55', 'real_56'],
    '7f': ['real_65']
}
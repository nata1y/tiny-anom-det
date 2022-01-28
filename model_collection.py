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
          # 'orion': (Orion(), [], []),
          # 'sr': (SpectralResidual, [Real(low=0.01, high=1.0001, name='THRESHOLD'),
          #                           Integer(low=1, high=30, name='MAG_WINDOW'),
          #                           Integer(low=5, high=1000, name='SCORE_WINDOW'),
          #                           Integer(low=1, high=100, name='sensitivity')],
          #        [THRESHOLD, MAG_WINDOW, SCORE_WINDOW, 99]),
          # 'lstm': (LSTM_autoencoder, [Real(low=0.5, high=3.0, name='threshold')], [1.5]),
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
    '1a_generated': ['art_daily_no_noise.csv', 'art_daily_perfect_square_wave.csv', 'art_daily_small_noise.csv',
                     'art_flatline.csv', 'art_noisy.csv', 'ec2_cpu_utilization_c6585a.csv', 'exchange-3_cpm_results.csv',
                     'occupancy_6005.csv', 'occupancy_t4013.csv', 'rds_cpu_utilization_cc0c53.csv',
                     'Twitter_volume_FB.csv', 'real_1.csv', 'real_10.csv', 'real_12.csv', 'real_16.csv', 'real_18.csv',
                     'real_21.csv', 'real_24.csv', 'real_25.csv', 'real_3.csv', 'real_30.csv', 'real_33.csv',
                     'real_35.csv', 'real_36.csv', 'real_4.csv', 'real_41.csv', 'real_42.csv', 'real_45.csv',
                     'real_49.csv', 'real_5.csv', 'real_50.csv', 'real_53.csv', 'real_54.csv', 'real_58.csv',
                     'real_59.csv', 'real_62.csv', 'real_64.csv', 'real_67.csv', 'real_8.csv', 'synthetic_100.csv',
                     'synthetic_11.csv', 'synthetic_13.csv', 'synthetic_14.csv', 'synthetic_19.csv', 'synthetic_21.csv',
                     'synthetic_23.csv', 'synthetic_25.csv', 'synthetic_27.csv', 'synthetic_28.csv', 'synthetic_30.csv',
                     'synthetic_31.csv', 'synthetic_33.csv', 'synthetic_34.csv', 'synthetic_35.csv', 'synthetic_37.csv',
                     'synthetic_39.csv', 'synthetic_41.csv', 'synthetic_42.csv', 'synthetic_43.csv', 'synthetic_44.csv',
                     'synthetic_46.csv', 'synthetic_47.csv', 'synthetic_48.csv', 'synthetic_49.csv', 'synthetic_5.csv',
                     'synthetic_51.csv', 'synthetic_53.csv', 'synthetic_55.csv', 'synthetic_56.csv', 'synthetic_57.csv',
                     'synthetic_58.csv', 'synthetic_59.csv', 'synthetic_60.csv', 'synthetic_61.csv', 'synthetic_62.csv',
                     'synthetic_63.csv', 'synthetic_65.csv', 'synthetic_67.csv', 'synthetic_69.csv', 'synthetic_7.csv',
                     'synthetic_70.csv', 'synthetic_71.csv', 'synthetic_72.csv', 'synthetic_73.csv', 'synthetic_74.csv',
                     'synthetic_75.csv', 'synthetic_76.csv', 'synthetic_77.csv', 'synthetic_79.csv', 'synthetic_81.csv',
                     'synthetic_83.csv', 'synthetic_84.csv', 'synthetic_85.csv', 'synthetic_86.csv', 'synthetic_87.csv',
                     'synthetic_88.csv', 'synthetic_89.csv', 'synthetic_9.csv', 'synthetic_90.csv', 'synthetic_91.csv',
                     'synthetic_93.csv', 'synthetic_95.csv', 'synthetic_97.csv', 'synthetic_98.csv', 'synthetic_99.csv',
                     'A3Benchmark-TS64.csv', 'A3Benchmark-TS66.csv'],
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
import os
import pandas as pd

from models.sr.spectral_residual import SpectralResidual, THRESHOLD, MAG_WINDOW, SCORE_WINDOW, DetectMode


def detect_anomaly(series, threshold, mag_window, score_window, sensitivity, detect_mode):
    detector = SpectralResidual(series=series, threshold=threshold, mag_window=mag_window, score_window=score_window,
                                sensitivity=sensitivity, detect_mode=detect_mode)
    print(detector.detect())


# if __name__ == '__main__':
#     root_path = os.getcwd()
#     dataset, type = 'yahoo', 'synthetic'
#     train_data_path = root_path + '/datasets/' + dataset + '/' + type + '/'
#     for filename in os.listdir(train_data_path):
#         f = os.path.join(train_data_path, filename)
#         if os.path.isfile(f):
#             data = pd.read_csv(f)
#
#             detect_anomaly(data, THRESHOLD, MAG_WINDOW, SCORE_WINDOW, 99, DetectMode.anomaly_only)

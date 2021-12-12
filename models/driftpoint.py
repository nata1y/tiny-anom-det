import datetime
import pandas as pd


class DriftPointModel:
    drift_now = False

    def __init__(self, dataset, filename, driftdetector, pointdetector):
        self.dataset = dataset
        self.filename = filename
        self.drift_detector = driftdetector
        self.anomaly_detector = pointdetector
        self.drift_windows = []
        self.detcted_anomalies = []

    def fit(self, X):
        self.anomaly_detector.fit(X, self.dataset)

    def predict(self, X):
        x, ts = X['value'].tolist()[-1], X['timestamp'].tolist()[-1]
        self.drift_detector.add_element(x)
        anomaly_index = self.anomaly_detector.predict(X)[-1]

        if self.drift_detector.detected_change():
            self.drift_detector.reset()
            # self.anomaly_detector.retrain()
            try:
                self.drift_windows.append((datetime.datetime.strptime(ts, '%m/%d/%Y %H:%M'),
                                      datetime.datetime.strptime(ts, '%m/%d/%Y %H:%M')))
            except:
                try:
                    self.drift_windows.append((datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S'),
                                          datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')))
                except:
                    self.drift_windows.append((ts, ts))
            drift_index = 1
        else:
            drift_index = 0

        if self.drift_now:
            self.detcted_anomalies.append(ts)
            return [1]

        if drift_index == anomaly_index:
            if drift_index == 1:
                self.drift_now = not self.drift_now
                self.detcted_anomalies.append(ts)
            return [drift_index]
        else:
            return [0]

    def plot(self, data_test_snippest, type, data_test):
        self.anomaly_detector.plot_ensemble(data_test_snippest, self.dataset, type, 'drift_' + self.filename, data_test,
                                   self.drift_windows, self.detcted_anomalies)

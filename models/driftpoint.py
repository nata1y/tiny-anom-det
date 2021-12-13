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
        self.detected_anomalies = []
        self.idx_drift_detected = None
        self.point_drift_alarming = 0
        self.latency_before_retrain = 50

    def fit(self, X):
        self.anomaly_detector.fit(X, self.dataset)

    def predict(self, X):
        group_anomalies = []
        for point in range(X.shape[0]):
            x, ts = X['value'].tolist()[point], X['timestamp'].tolist()[point]
            self.drift_detector.add_element(x)

            if self.drift_detector.detected_change():
                group_anomalies.append(1)
                try:
                    self.drift_windows.append((datetime.datetime.strptime(ts, '%m/%d/%Y %H:%M'),
                                               datetime.datetime.strptime(ts, '%m/%d/%Y %H:%M')))
                except:
                    try:
                        self.drift_windows.append((datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S'),
                                                   datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')))
                    except:
                        self.drift_windows.append((ts, ts))
            else:
                group_anomalies.append(0)

        point_anomalies = self.anomaly_detector.predict(X)
        y_pred = []
        if sum(point_anomalies) == len(point_anomalies):
            y_pred = [1 for _ in range(len(point_anomalies))]
            self.point_drift_alarming += len(point_anomalies)
            self.detected_anomalies += X.index.tolist()

        else:
            for pa, ga, ts in zip(point_anomalies, group_anomalies, X.index.tolist()):
                if pa == ga == 1:
                    self.detected_anomalies.append(ts)
                    y_pred.append(1)
                else:
                    y_pred.append(0)

        return y_pred

    def plot(self, data_test_snippest, type, data_test):
        self.anomaly_detector.plot_ensemble(data_test_snippest, self.dataset, type, 'drift_' + self.filename, data_test,
                                   self.drift_windows, self.detected_anomalies)

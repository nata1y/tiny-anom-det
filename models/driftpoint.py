import copy
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
        self.anomaly_in_memory = []

        self.drift_now = False

        self.anomaly_interval_cutoff = 5

    def _longest_anomaly(self, s):
        ans, temp = 0, 0
        idx, candidate_idx = 0, -1
        for i in range(1, len(s)):
            if s[i] == s[i - 1] == 1:
                temp += 1
                if candidate_idx != -1:
                    candidate_idx = i - 1
            else:
                ans = max(ans, temp)
                if ans == temp and ans != 0:
                    idx = candidate_idx
                temp = 1
                candidate_idx = -1

        ans = max(ans, temp)

        return ans, idx

    def _join_anomalies(self, new_anomalies):
        for idx, (existing, new) in enumerate(zip(self.anomaly_in_memory, new_anomalies[:len(self.anomaly_in_memory)])):
            self.anomaly_in_memory[idx] = existing * new

        self.anomaly_in_memory += new_anomalies[len(self.anomaly_in_memory):]

    def fit(self, X):
        self.anomaly_detector.fit(X, self.dataset)

    def predict(self, X,  optimization=False):
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

        point_anomalies = self.anomaly_detector.predict(X, in_dp=False, optimization=optimization)
        y_pred = []
        for idx in range(X.shape[0]):
            if group_anomalies[idx] == 1:
                y_pred.append(1)
                self.drift_now = not self.drift_now
            else:
                if self.drift_now:
                    y_pred.append(1)
                else:
                    y_pred.append(point_anomalies[idx])

            if y_pred[idx] == 1:
                self.detected_anomalies.append(X.index.tolist()[idx])

        return y_pred

    def predict_one_step(self, X):
        group_anomalies = []
        in_window_detector = copy.deepcopy(self.drift_detector)
        for point in range(X.shape[0]):
            x, ts = X['value'].tolist()[point], X['timestamp'].tolist()[point]
            in_window_detector.add_element(x)

            if in_window_detector.detected_change():
                group_anomalies.append(1)
            else:
                group_anomalies.append(0)

        point_anomalies = self.anomaly_detector.predict(X, in_dp=True)
        if not self.anomaly_in_memory:
            self.anomaly_in_memory = point_anomalies
        else:
            self.anomaly_in_memory = self.anomaly_in_memory[1:]
            self._join_anomalies(point_anomalies)

        for idx, (pa, ga) in enumerate(zip(point_anomalies, group_anomalies)):
            if ga == 1:
                if not self.drift_now:
                    self.idx_stop_froze = None
                    self.idx_start_froze = idx
                else:
                    self.idx_stop_froze = idx
                self.drift_now = not self.drift_now
                period_len, start_idx = self._longest_anomaly(self.anomaly_in_memory[:idx+1])
                if period_len >= self.anomaly_interval_cutoff:
                    for i in range(start_idx, idx):
                        self.anomaly_in_memory[i] = 1 if self.drift_now else 0
            if self.drift_now:
                self.anomaly_in_memory[idx] = 1
            else:
                self.anomaly_in_memory[idx] = 0

        if self.anomaly_in_memory[0] == 1:
            self.detected_anomalies.append(X.index.tolist()[0])

        self.drift_detector.add_element(X['value'].tolist()[0])
        if self.drift_detector.detected_change():
            ts = X.index.tolist()[0]
            try:
                self.drift_windows.append((datetime.datetime.strptime(ts, '%m/%d/%Y %H:%M'),
                                           datetime.datetime.strptime(ts, '%m/%d/%Y %H:%M')))
            except:
                try:
                    self.drift_windows.append((datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S'),
                                               datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')))
                except:
                    self.drift_windows.append((ts, ts))

        return [self.anomaly_in_memory[0]]

    def plot(self, data_test_snippest, type, data_test):
        self.anomaly_detector.plot_ensemble(data_test_snippest, self.dataset, type, 'drift_' + self.filename, data_test,
                                   self.drift_windows, self.detected_anomalies)

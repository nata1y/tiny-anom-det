
class DriftDetectorWrapper:

    def __init__(self, detector):
        self.detector = detector()
        self.nodrift = "No"
        self.alert = "Alert"
        self.drift = "Drift"
        self.sensor_drift = False

    def record(self, error):
        '''
        Record error concept
        '''
        for e in error:
            self.detector.add_element(e)

    def update(self, error, t):
        '''
        method to update ewma with error at time t
        '''
        self.detector.add_element(error)
        self.sensor_drift = False

    def monitor(self):
        '''
        method to monitor the condition of the detector
        '''

        if self.detector.detected_change():
            self.sensor_drift = True
            return self.drift
        elif self.detector.detected_warning_zone():
            return self.alert
        else:
            return self.nodrift

    def reset(self):
        '''
        reset drift detector to initial state
        '''
        self.detector.reset()
        self.sensor_drift = True

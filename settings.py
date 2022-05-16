test_drifts = False
anomaly_window = 65
step = 65
entropy_params = {'kpi_train': {'factor': 2.5, 'window': 35},
                  'NAB_windows': {'factor': 2.9, 'window': 65},
                  'yahoo_real': {'factor': 1.0, 'window': 85},
                  'yahoo_synthetic': {'factor': 1.8, 'window': 25}
                  }

data_in_memory_sz = 3000

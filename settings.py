test_drifts = False
anomaly_window = 65
step = 65
entropy_params = {'kpi_train': {'factor': 2.0, 'window': 35},
                  'NAB_windows': {'factor': 2.9, 'window': 65},
                  'yahoo_A1Benchmark': {'factor': 1.0, 'window': 85},
                  'yahoo_A2Benchmark': {'factor': 1.5, 'window': 60},
                  'yahoo_A3Benchmark': {'factor': 1.5, 'window': 60},
                  'yahoo_A4Benchmark': {'factor': 1.5, 'window': 60}
                  }

data_in_memory_sz = 3000
no_optimization = True
mi = 100

#parametros IDPSO
it = 50
inercia_inicial = 0.8
inercia_final = 0.4
xmax = 1
c1 = 2
c2 = 2
crit = 2
split_dataset = [0.8, 0.2, 0]

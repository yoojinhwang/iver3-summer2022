import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def regression(x, y):
    return np.linalg.lstsq(x[:, np.newaxis], y, rcond=None)[0]

data = pd.read_csv('./data/tag77-0m-air-test-0.csv')
distances = np.array(data['total_distance'])
times = np.array(data['total_dt'])
dt = np.diff(times)
n = np.round(dt / np.min(dt))
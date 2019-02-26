import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y

plt.rcParams["figure.figsize"] = (20,3)

csv_file = "./Data/chicago-complete.weekly.2018-12-03-to-2018-12-09/data.csv"
df = pd.read_csv(csv_file)

print("Values read")

sensors_list = ["co", "h2s", "no2", "so2", "o3"]

sensor = df[df["sensor"] == "no2"]
unique_nodes = list(sensor.node_id.unique())

node = sensor[sensor["node_id"] == unique_nodes[1]]  # "001e0610f05c"]
print("Values filtered")
print(unique_nodes[0])
assert len(node) > 1000

value_raw = np.array(list(float(v) for v in node.value_raw))
value_hrf = np.array(list(float(v) for v in node.value_hrf))

plt.subplot(211); plt.plot(smooth(value_hrf), "b-*"); plt.subplot(212); plt.plot(smooth(value_raw), "r-*"); plt.show()
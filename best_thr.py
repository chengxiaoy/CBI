import joblib

metric = joblib.load('checkpoints/exp_10_index_0_metric_data.pkl')
val_data = metric[25]['val']
import numpy as np
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

thr_list = np.arange(0.3, 0.8, 0.05)
f1_list = []

for thr in thr_list:
    a = np.zeros(val_data[0].shape)
    a[val_data[1] >= thr] = 1
    f1_s = f1_score(val_data[0], a, average='micro')
    f1_list.append(f1_s)

plt.plot(thr_list, f1_list)
plt.show()

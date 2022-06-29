import os
import numpy as np

kvalue = 'K10'
path = './Experiments/MaFengWo/AGREE/data/GRADI_100/'

filename = 'EVA_group_user_E32_batch128_topK10_drop_ratio_0.20_lambda_0.50_eta_1.00_lr_0.000005_'

def result(path, filename):
    results = []
    for i in range(5):
        d = np.loadtxt(os.path.join(path, filename + str(i)))
        data = d[-1,:]
        results.append(data)
    return np.array(results)

def stadata(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    result = np.vstack((mean, std))
    return result

data = result(path, filename)
sta = stadata(data)

print('save mean data...')
np.savetxt(os.path.join(path, 'gradi' + kvalue), data, fmt='%1.4f', delimiter=' ')
np.savetxt(os.path.join(path, 'gradista' + kvalue), sta, fmt='%1.4f', delimiter=' ')

print('Everything is OK!')
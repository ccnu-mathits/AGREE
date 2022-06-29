import os
import numpy as np

kvalue = '10'
dataname = 'MaFengWo'
path = '/home/admin123/ruxia/HAN-CDGRccnu/Experiments/AGREE/data/'
dir_name = os.path.join(path, dataname) + '/results'
filename = 'EVA_group_user_E32_batch128_topK10_drop_ratio_0.20_lambda_0.50_eta_1.00_lr_0.000200_'
filename = os.path.join(dir_name, filename)

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
np.savetxt(os.path.join(dir_name, 'agree' + kvalue), data, fmt='%1.4f', delimiter=' ')
np.savetxt(os.path.join(dir_name, 'agree' + 'sta' + kvalue), sta, fmt='%1.4f', delimiter=' ')

print('Everything is alright!')
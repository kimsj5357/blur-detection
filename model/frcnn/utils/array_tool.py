import numpy as np
import torch as t

def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, t.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, list):
        return np.array(data)

def totensor(data):
    if isinstance(data, np.ndarray):
        return t.from_numpy(data).cuda()
    if isinstance(data, t.Tensor):
        return data.cuda()
    if isinstance(data, list):
        return t.tensor(data)

def todict(data):
    for key, item in data.items():
        data[key] = tonumpy(item)
        if len(data[key]) == 1:
            data[key] = data[key][0]
    return data
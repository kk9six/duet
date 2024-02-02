import torch
import os
import pickle
import yaml

def count(X, single_dim=1):
    '''
        count batch number and sample number of input
    '''
    dim = X.dim()
    if dim == single_dim:
        return 0, 0
    elif dim == single_dim + 1:
        return 0, X.shape[0]
    elif dim == single_dim + 2:
        return X.shape[0], X.shape[1]
    else:
        raise 'Unknown shape.'
        
def btmm(mat1, mat2):
    return torch.einsum("bji, bjk -> bik", mat1, mat2)

def btmv(a, b):
    """
    batch transposed matrix product vector
    """
    return torch.einsum('bij, bi -> bj', a, b)

def bmv(a, b):
    """
    matrix product vector
    """
    return torch.einsum('nij, nj -> ni', a, b)

def bbmv(a, b):
    return torch.einsum('bnij, bnj -> bni', a, b)

def isclose(x, y):
    return (x-y).abs() < 0.02


import os
import pandas as pd
def list_models():
    print(pd.DataFrame(os.listdir('./models'), columns=['model']))


def pload(*f_names):
    """Pickle load"""
    f_name = os.path.join(*f_names)
    with open(f_name, "rb") as f:
        pickle_dict = pickle.load(f)
    return pickle_dict

def pdump(pickle_dict, *f_names):
    """Pickle dump"""
    f_name = os.path.join(*f_names)
    with open(f_name, "wb") as f:
        pickle.dump(pickle_dict, f)

def mkdir(*paths):
    '''Create a directory if not existing.'''
    path = os.path.join(*paths)
    if not os.path.exists(path):
        os.mkdir(path)

def yload(*f_names):
    """YAML load"""
    f_name = os.path.join(*f_names)
    with open(f_name, 'r') as f:
        yaml_dict = yaml.load(f)
    return yaml_dict

def ydump(yaml_dict, *f_names):
    """YAML dump"""
    f_name = os.path.join(*f_names)
    with open(f_name, 'w') as f:
        yaml.dump(yaml_dict, f, default_flow_style=False)

def bmtv(mat, vec):
    """batch matrix transpose vector product"""
    return torch.einsum('bji, bj -> bi', mat, vec)

def bmtm(mat1, mat2):
    """batch matrix transpose matrix product"""
    return torch.einsum("bji, bjk -> bik", mat1, mat2)

def bmmt(mat1, mat2):
    """batch matrix matrix transpose product"""
    return torch.einsum("bij, bkj -> bik", mat1, mat2)

def try_gpu(i=0):
    if num_gpus() >= i+1:
        return gpu(i)
    return cpu()

def try_all_gpus():
    return [gpu(i) for i in range(num_gpus())]

def cpu():
    return torch.device('cpu')

def gpu(i=0):
    return torch.device(f'cuda:{i}')

def num_gpus():
    return torch.cuda.device_count()


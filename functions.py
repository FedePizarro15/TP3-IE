import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tp3 import CONFIG_1
from utils import *

P = (100, 50)

def solve_ls(config: dict[str, dict]):
    h_rows = []
    z_rows = []

    for _, value in config.items():
        xi, yi = value['pos']
        thetai = value['angulo']
                
        h_i = [1, -np.tan(thetai)]
        h_rows.append(h_i)
        
        z_i = xi - yi * np.tan(thetai)
        z_rows.append(z_i)

    h = np.array(h_rows) # Dimensión (M, 2)
    z = np.array(z_rows) # Dimensión (M, 1)

    h_T = h.T
    
    xtx = h_T @ h 
    
    xtx_inv = np.linalg.inv(xtx)
    
    p_hat = xtx_inv @ h_T @ z

    p_hat = np.round(p_hat, 2)

    return tuple(p_hat)
    
def get_data(config: dict[str, dict]):
    data = {}
    
    for key, value in config.items():
        pos = value['pos']
        data[key] = pos
    
    df = pd.DataFrame.from_dict(data, orient='index', columns=['x', 'y'])
    df.index.name = 'Positions'
    
    return df

def plot_prediction(config: dict[str, dict], true_pos: tuple[float, float] = P):
    x_p, y_p = solve_ls(config)
    x_p, y_p = float(x_p), float(y_p)
    
    display_text(f'Referencias: CONFIG_1, P: {true_pos}, Predicciones: {x_p, y_p}', level=3)
    
    data = get_data(config)
    
    x = data['x']
    y = data['y']
    
    plt.scatter(x=x, y=y, c='blue', marker='o', s=60, label='Referencias')
    plt.scatter(x=true_pos[0], y=true_pos[1], c='black', marker='o', s=100, label='Real (P)')
    plt.scatter(x=x_p, y=y_p, c='red', marker='x', s=80, linewidths=2, label='Estimación (LS)')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.show()
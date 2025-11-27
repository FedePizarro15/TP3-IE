import numpy as np
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse

from tp3 import CONFIG_1, CONFIG_2, CONFIG_3, CONFIG_4
from utils import *

config = dict[str, dict]

def get_data(config: config):
    x = np.array([v['pos'][0] for v in config.values()])
    y = np.array([v['pos'][1] for v in config.values()])
    theta = np.array([v['angulo'] for v in config.values()])
    
    return x, y, theta

def solve_ls(x: list[float], y: list[float], theta: list[float]) -> tuple[float, float]:
    x_rows = []
    y_rows = []

    for x_i, y_i, theta_i in zip(x, y, theta):
        x_rows.append([1, -np.tan(theta_i)])
        y_rows.append(x_i - y_i * np.tan(theta_i))

    X = np.array(x_rows) # Dimensión (M, 2)
    Y = np.array(y_rows) # Dimensión (M, 1)

    X_t = X.T
    
    X_t_X = X_t @ X 
    
    X_t_X_inv = np.linalg.inv(X_t_X)
    
    beta = X_t_X_inv @ X_t @ Y

    return tuple(beta)
        
def condition_number(matrix: np.ndarray):
    matrix_inv = np.linalg.pinv(matrix)
    
    norm_matrix = np.linalg.norm(matrix, ord=2)
    norm_matrix_inv = np.linalg.norm(matrix_inv, ord=2)
    
    return norm_matrix * norm_matrix_inv
    
def setup_base_plot(ax: plt.Axes, x_ref, y_ref, true_pos: tuple[float, float] = (100, 50)) -> None:
    ax.scatter(x_ref, y_ref, c='blue', marker='o', s=60, label='Referencias')
    ax.scatter(true_pos[0], true_pos[1], c='black', marker='o', s=100, label='Real (P)')
    
    ax.set_xlabel('Coordenada x')
    ax.set_ylabel('Coordenada y')
    
    ax.grid(True, linestyle='--', alpha=0.6)

def estimate_position(config: config = CONFIG_1, pos: tuple[float, float] = (100, 50)) -> None:
    x_true, y_true = pos[0], pos[1]
    
    x_ref, y_ref, thetas = get_data(config)
    
    x_pred, y_pred = solve_ls(x_ref, y_ref, thetas)
    
    plt.scatter(x_ref, y_ref, c='blue', marker='o', s=60, label='Referencias')
    plt.scatter(x_true, y_true, c='black', marker='o', s=100, label='Real (P)')
    plt.scatter(x_pred, y_pred, c='red', marker='x', s=80, linewidths=2, label='Estimación (LS)')
    
    x_pred, y_pred = np.round((x_pred, y_pred), 2).tolist()

    display_text('Real')
    display_text(f'{round(float(x_true), 2), round(float(y_true), 2)}', level=3)
    display_text(f'Predicción')
    display_text(f'{x_pred, y_pred}', level=3)

    plt.xlabel('Coordenada x')
    plt.ylabel('Coordenada y')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.show()
    
    return x_pred, y_pred

def noisy_estimations(config: config = CONFIG_1, variances: list[int] = [4, 25, 100], n: int = 50) -> None:
    if len(variances) > 1:
        display_text(f'{n} Estimaciones con Ruido - (Δx, Δy ∼ N(0, σ²))')
            
    x_ref, y_ref, thetas = get_data(config)
    
    for var in variances:
        if len(variances) > 1:
            display_text(f'σ² = {var}', level=3)
        
        sigma = np.sqrt(var) # Desviación estándar
        
        x_predictions = []
        y_predictions = []
        
        for _ in range(n):            
            x_noise = np.random.normal(0, sigma, size=len(x_ref))
            y_noise = np.random.normal(0, sigma, size=len(y_ref))
            
            x_ref_noisy = x_ref + x_noise
            y_ref_noisy = y_ref + y_noise
            
            x_pred, y_pred = solve_ls(x_ref_noisy, y_ref_noisy, thetas)
            
            x_predictions.append(x_pred)
            y_predictions.append(y_pred)
    
        fig, ax = plt.subplots(figsize=(8, 6))
        
        setup_base_plot(ax, x_ref, y_ref)
        
        ax.scatter(x_predictions, y_predictions, c='red', marker='x', s=80, linewidths=2, label='Estimación (LS)', alpha=0.5)
        
        draw_confidence_ellipse(ax, x_predictions, y_predictions, n_std=2.0)
                
        ax.legend()
                
        ax.legend()
        plt.show()
        
def estimate_positions(configs: list[config] = [CONFIG_2, CONFIG_3, CONFIG_4], variance: int = 4, n: int = 50):
    display_text(f'{n} Estimaciones con Ruido - (Δx, Δy ∼ N(0, {variance}))')

    for i, config in enumerate(configs):        
        _, _, thetas = get_data(config)
        h = np.column_stack((np.ones(len(thetas)), -np.tan(thetas)))
        k = condition_number(h).round(2)
        
        if k > 10:
            display_text(f'config_{i + 2} - κ = {k} > 10 - Mal condicionada', level=3)
        else:
            display_text(f'config_{i + 2} - κ = {k} < 10 - Bien condicionada', level=3)
        
        noisy_estimations(config, [variance])

def draw_confidence_ellipse(ax: plt.Axes, x_data: list[float], y_data: list[float], n_std: float = 2.0):
    if len(x_data) < 2: return

    cov = np.cov(x_data, y_data)
    
    vals, vecs = np.linalg.eigh(cov)
    
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    
    width, height = 2 * n_std * np.sqrt(vals)
    
    ellipse = Ellipse(xy=(np.mean(x_data), np.mean(y_data)),
                      width=width, height=height,
                      angle=theta, edgecolor='green', facecolor='none', linestyle='--',
                      label='Confidence Ellipse (95%)', linewidth=2, alpha=0.8, zorder=0)
    
    ax.add_patch(ellipse)
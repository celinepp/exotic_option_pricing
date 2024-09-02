import logging
import time
from functools import wraps

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_input_parameters(**kwargs):
    for param, value in kwargs.items():
        if value is None:
            raise ValueError(f"Parameter '{param}' cannot be None.")
        if isinstance(value, (int, float)):
            if np.isnan(value) or np.isinf(value):
                raise ValueError(f"Parameter '{param}' must be a finite number.")
            if param in ['S0', 'K', 'T', 'sigma', 'B'] and value <= 0:
                raise ValueError(f"Parameter '{param}' must be positive.")
        if isinstance(value, int):
            if value <= 0:
                raise ValueError(f"Parameter '{param}' must be a positive integer.")
    logger.info("All input parameters are valid.")

def calculate_discount_factor(r, T):
    return np.exp(-r * T)

def log_execution_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting '{func.__name__}'...")
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"Finished '{func.__name__}' in {end_time - start_time:.4f} seconds.")
        return result
    return wrapper

def finite_difference_grid(x_min, x_max, Nspace):
    x, dx = np.linspace(x_min, x_max, Nspace, retstep=True)
    return x, dx

def finite_difference_time_grid(T, Ntime):
    t, dt = np.linspace(0, T, Ntime, retstep=True)
    return t, dt

def tridiagonal_matrix(a, b, c, size):
    from scipy import sparse
    diagonals = [a, b, c]
    offsets = [-1, 0, 1]
    return sparse.diags(diagonals, offsets, shape=(size, size), format='csc')

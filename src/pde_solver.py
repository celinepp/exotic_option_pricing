
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu, spsolve
from utils import validate_input_parameters, finite_difference_grid, finite_difference_time_grid, tridiagonal_matrix

# -----------------------------------------
# PDE Solver for Barrier Options
# -----------------------------------------
def pde_barrier_option(S0, K, T, r, sigma, B, Nspace=14000, Ntime=10000):
    """
    Solves the PDE for an up-and-out barrier option.

    Parameters:
    S0 (float): Initial stock price
    K (float): Strike price
    T (float): Time to maturity
    r (float): Risk-free interest rate
    sigma (float): Volatility of the underlying asset
    B (float): Barrier level
    Nspace (int): Number of space steps
    Ntime (int): Number of time steps

    Returns:
    float: Estimated price of the barrier option.
    """
    # Validate input parameters
    validate_input_parameters(S0=S0, K=K, T=T, r=r, sigma=sigma, B=B, Nspace=Nspace, Ntime=Ntime)
    
    X0 = np.log(S0)
    S_max = B
    S_min = float(K) / 3
    x_max = np.log(S_max)
    x_min = np.log(S_min)

    # Generate finite difference grids
    x, dx = finite_difference_grid(x_min, x_max, Nspace)
    T_array, dt = finite_difference_time_grid(T, Ntime)

    # Payoff function and boundary conditions
    Payoff = np.maximum(np.exp(x) - K, 0)
    V = np.zeros((Nspace, Ntime))
    offset = np.zeros(Nspace - 2)
    V[:, -1] = Payoff
    V[-1, :] = 0
    V[0, :] = 0

    # Coefficients for the tridiagonal matrix
    sig2 = sigma * sigma
    dxx = dx * dx
    a = (dt / 2) * ((r - 0.5 * sig2) / dx - sig2 / dxx) * np.ones(Nspace - 2)
    b = (1 + dt * (sig2 / dxx + r)) * np.ones(Nspace - 2)
    c = -(dt / 2) * ((r - 0.5 * sig2) / dx + sig2 / dxx) * np.ones(Nspace - 2)

    # Construct and solve tridiagonal system
    D = tridiagonal_matrix(a[1:], b, c[:-1], size=Nspace-2)
    DD = splu(D)

    for i in range(Ntime - 2, -1, -1):
        offset[0] = a[1] * V[0, i]
        offset[-1] = c[-1] * V[-1, i]
        V[1:-1, i] = DD.solve(V[1:-1, i + 1] - offset)

    oPrice = np.interp(X0, x, V[:, 0])
    return oPrice
# -----------------------------------------
# PDE Solver for Down-and-In Barrier Options
# -----------------------------------------
def pde_down_in_option(S0, K, T, r, sigma, B, Nspace=14000, Ntime=10000):
    """
    Solves the PDE for a down-and-in barrier option.

    Parameters:
    S0 (float): Initial stock price
    K (float): Strike price
    T (float): Time to maturity
    r (float): Risk-free interest rate
    sigma (float): Volatility of the underlying asset
    B (float): Barrier level
    Nspace (int): Number of space steps
    Ntime (int): Number of time steps

    Returns:
    float: Estimated price of the down-and-in barrier option.
    """
    # Validate input parameters
    validate_input_parameters(S0=S0, K=K, T=T, r=r, sigma=sigma, B=B, Nspace=Nspace, Ntime=Ntime)
    
    X0 = np.log(S0)
    S_max = K * 3  # The max of S corresponds to the strike price
    S_min = B      # The min of S corresponds to the barrier level
    x_max = np.log(S_max)
    x_min = np.log(S_min)

    # Generate finite difference grids
    x, dx = finite_difference_grid(x_min, x_max, Nspace)
    T_array, dt = finite_difference_time_grid(T, Ntime)

    # Payoff function and boundary conditions
    Payoff = np.maximum(np.exp(x) - K, 0)
    V = np.zeros((Nspace, Ntime))
    offset = np.zeros(Nspace - 2)
    V[:, -1] = Payoff
    V[-1, :] = np.exp(x_max) - K * np.exp(-r * T_array[::-1])
    V[0, :] = 0

    # Coefficients for the tridiagonal matrix
    sig2 = sigma * sigma
    dxx = dx * dx
    a = (dt / 2) * ((r - 0.5 * sig2) / dx - sig2 / dxx) * np.ones(Nspace - 2)
    b = (1 + dt * (sig2 / dxx + r)) * np.ones(Nspace - 2)
    c = -(dt / 2) * ((r - 0.5 * sig2) / dx + sig2 / dxx) * np.ones(Nspace - 2)

    # Construct and solve tridiagonal system
    D = tridiagonal_matrix(a[1:], b, c[:-1], size=Nspace-2)
    DD = splu(D)

    for i in range(Ntime - 2, -1, -1):
        offset[0] = a[1] * V[0, i]
        offset[-1] = c[-1] * V[-1, i]
        V[1:-1, i] = DD.solve(V[1:-1, i + 1] - offset)

    oPrice = np.interp(X0, x, V[:, 0])
    return oPrice

# -----------------------------------------
# PDE Solver for Fixed Strike Asian Options
# -----------------------------------------
def pde_fixed_strike_asian_option(S0, K, T, r, sigma, Nspace=6000, Ntime=6000):
    """
    Solves the PDE for a fixed strike Asian option.

    Parameters:
    S0 (float): Initial stock price
    K (float): Strike price
    T (float): Time to maturity
    r (float): Risk-free interest rate
    sigma (float): Volatility of the underlying asset
    Nspace (int): Number of space steps
    Ntime (int): Number of time steps

    Returns:
    float: Estimated price of the fixed strike Asian option.
    """
    # Validate input parameters
    validate_input_parameters(S0=S0, K=K, T=T, r=r, sigma=sigma, Nspace=Nspace, Ntime=Ntime)
    
    def gamma(t):
        return 1 / (r * T) * (1 - np.exp(-r * (T - t)))

    def get_X0(S0, K, r, T):
        """Compute the variable x defined for the fixed strike Asian option."""
        return gamma(0) * S0 - np.exp(-r * T) * K

    y_max = 60
    y_min = -60

    # Generate finite difference grids
    y, dy = finite_difference_grid(y_min, y_max, Nspace)
    T_array, dt = finite_difference_time_grid(T, Ntime)

    # Payoff function and boundary conditions
    Payoff = np.maximum(y, 0)
    G = np.zeros((Nspace, Ntime))
    offset = np.zeros(Nspace - 2)
    G[:, -1] = Payoff
    G[-1, :] = y_max
    G[0, :] = 0

    # Time-stepping PDE solver
    for n in range(Ntime - 2, -1, -1):
        sig2 = sigma * sigma
        dyy = dy * dy
        a = -0.5 * (dt / dyy) * sig2 * (gamma(T_array[n]) - y[1:-1]) ** 2
        b = 1 + (dt / dyy) * sig2 * (gamma(T_array[n]) - y[1:-1]) ** 2
        cM = a[-1]
        D = tridiagonal_matrix(a[1:], b, a[:-1], size=Nspace-2)
        offset[0] = a[1] * G[0, n]
        offset[-1] = cM * G[-1, n]
        G[1:-1, n] = spsolve(D, G[1:-1, n + 1] - offset)

    X0 = get_X0(S0, K, r, T)
    oPrice = S0 * np.interp(X0 / S0, y, G[:, 0])
    return oPrice

def pde_floating_strike_asian_option(S0, K, T, r, sigma, Nspace=4000, Ntime=7000):
    """
    Solves the PDE for a floating strike Asian option.

    Parameters:
    S0 (float): Initial stock price
    K (float): Strike price
    T (float): Time to maturity
    r (float): Risk-free interest rate
    sigma (float): Volatility of the underlying asset
    Nspace (int): Number of space steps
    Ntime (int): Number of time steps

    Returns:
    float: Estimated price of the floating strike Asian option.
    """
    # Validate input parameters
    validate_input_parameters(S0=S0, K=K, T=T, r=r, sigma=sigma, Nspace=Nspace, Ntime=Ntime)
    
    x_max = 10
    x_min = 0

    # Generate finite difference grids
    x, dx = finite_difference_grid(x_min, x_max, Nspace)
    T_array, dt = finite_difference_time_grid(T, Ntime)

    # Payoff function and boundary conditions
    Payoff = np.maximum(x - 1, 0)
    V = np.zeros((Nspace, Ntime))
    offset = np.zeros(Nspace - 2)
    V[:, -1] = Payoff
    V[-1, :] = x_max - 1  # Boundary condition at the maximum x value
    V[0, :] = 0  # Boundary condition at the minimum x value

    # Time-stepping PDE solver
    for n in range(Ntime - 2, -1, -1):
        sig2 = sigma * sigma
        dxx = dx * dx
        max_part = np.maximum(x[1:-1] * (r - (x[1:-1] - 1) / T_array[n]), 0)
        min_part = np.minimum(x[1:-1] * (r - (x[1:-1] - 1) / T_array[n]), 0)

        a = min_part * (dt / dx) - 0.5 * (dt / dxx) * sig2 * (x[1:-1]) ** 2
        b = 1 + dt * (r - (x[1:-1] - 1) / T_array[n]) + (dt / dxx) * sig2 * (x[1:-1]) ** 2 + dt / dx * (max_part - min_part)
        c = -max_part * (dt / dx) - 0.5 * (dt / dxx) * sig2 * (x[1:-1]) ** 2

        try:
            D = tridiagonal_matrix(a[1:], b, c[:-1], size=Nspace-2)
            offset[0] = a[1] * V[0, n]
            offset[-1] = c[-1] * V[-1, n]
            V[1:-1, n] = spsolve(D, V[1:-1, n + 1] - offset)
        except Exception as e:
            print(f"An error occurred at time step {n}: {e}")
            break

    oPrice = S0 * np.interp(1, x, V[:, 0])
    return oPrice



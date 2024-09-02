

from pde_solver import pde_barrier_option, pde_down_in_option

def price_up_and_out_barrier_option(S0, K, T, r, sigma, B, Nspace=14000, Ntime=10000):
    """
    Prices an up-and-out barrier option using the PDE method.

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
    float: Estimated price of the up-and-out barrier option.
    """
    return pde_barrier_option(S0, K, T, r, sigma, B, Nspace, Ntime)

def price_down_and_in_barrier_option(S0, K, T, r, sigma, B, Nspace=14000, Ntime=10000):
    """
    Prices a down-and-in barrier option using the PDE method.

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
    return pde_down_in_option(S0, K, T, r, sigma, B, Nspace, Ntime)

if __name__ == "__main__":
    # Example usage
    S0 = 100
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2
    B = 120

    price_up_out = price_up_and_out_barrier_option(S0, K, T, r, sigma, B)
    price_down_in = price_down_and_in_barrier_option(S0, K, T, r, sigma, B)

    print(f"Up and Out Barrier Option Price: {price_up_out}")
    print(f"Down and In Barrier Option Price: {price_down_in}")

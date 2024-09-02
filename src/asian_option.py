

from pde_solver import pde_fixed_strike_asian_option, pde_floating_strike_asian_option

def price_fixed_strike_asian_option(S0, K, T, r, sigma, Nspace=6000, Ntime=6000):
    """
    Prices a fixed strike Asian option using the PDE method.

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
    return pde_fixed_strike_asian_option(S0, K, T, r, sigma, Nspace, Ntime)

def price_floating_strike_asian_option(S0, K, T, r, sigma, Nspace=4000, Ntime=7000):
    """
    Prices a floating strike Asian option using the PDE method.

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
    return pde_floating_strike_asian_option(S0, K, T, r, sigma, Nspace, Ntime)

if __name__ == "__main__":
    # Example usage
    S0 = 100
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2

    price_fixed = price_fixed_strike_asian_option(S0, K, T, r, sigma)
    price_floating = price_floating_strike_asian_option(S0, K, T, r, sigma)

    print(f"Fixed Strike Asian Option Price: {price_fixed}")
    print(f"Floating Strike Asian Option Price: {price_floating}")

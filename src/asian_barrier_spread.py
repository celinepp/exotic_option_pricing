
from pde_solver import pde_fixed_strike_asian_option, pde_barrier_option

def price_asian_barrier_spread(S0, K, T, r, sigma, B, Nspace_asian=6000, Ntime_asian=6000, Nspace_barrier=14000, Ntime_barrier=10000):
    """
    Prices an Asian-Barrier Spread option using a combination of PDE methods.

    Parameters:
    S0 (float): Initial stock price
    K (float): Strike price
    T (float): Time to maturity
    r (float): Risk-free interest rate
    sigma (float): Volatility of the underlying asset
    B (float): Barrier level
    Nspace_asian (int): Number of space steps for the Asian option component
    Ntime_asian (int): Number of time steps for the Asian option component
    Nspace_barrier (int): Number of space steps for the barrier option component
    Ntime_barrier (int): Number of time steps for the barrier option component

    Returns:
    float: Estimated price of the Asian-Barrier Spread option.
    """
    # Step 1: Price the Asian option component
    price_asian = pde_fixed_strike_asian_option(S0, K, T, r, sigma, Nspace_asian, Ntime_asian)

    # Step 2: Price the barrier option component
    price_barrier = pde_barrier_option(S0, K, T, r, sigma, B, Nspace_barrier, Ntime_barrier)

    # Step 3: Combine the results (this could be a simple sum or a more complex combination depending on the structure)
    spread_price = price_asian + price_barrier  # Adjust this combination logic as needed

    return spread_price

if __name__ == "__main__":
    # Example usage
    S0 = 100
    K = 100
    T = 1.0
    r = 0.05
    sigma = 0.2
    B = 120

    price_spread = price_asian_barrier_spread(S0, K, T, r, sigma, B)

    print(f"Asian-Barrier Spread Option Price: {price_spread}")

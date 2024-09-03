import numpy as np
import pandas as pd
from scipy.stats import norm
from utils import validate_input_parameters

def historical_simulation_asian_option(data, strike_price, r=0.01, T=1.0, option_type="call"):
    """
    Historical simulation for pricing a fixed strike Asian option.
    """
    validate_input_parameters(strike_price=strike_price, r=r, T=T)
    
    # Calculate the average price over the period
    average_price = data['Close'].mean()
    
    # Payoff calculation
    if option_type == "call":
        payoff = np.maximum(average_price - strike_price, 0)
    else:
        payoff = np.maximum(strike_price - average_price, 0)
    
    # Discounted payoff
    option_price = np.exp(-r * T) * payoff
    return option_price


def historical_simulation_barrier_option(data, strike_price, barrier_level, r=0.01, T=1.0, option_type="call"):
    """
    Historical simulation for pricing a Barrier option (e.g., up-and-out call).
    """
    validate_input_parameters(strike_price=strike_price, r=r, T=T, barrier_level=barrier_level)
    
    breached = data['Close'].max() > barrier_level

    if not breached:
        if option_type == "call":
            payoff = np.maximum(data['Close'].iloc[-1] - strike_price, 0)
        else:
            payoff = np.maximum(strike_price - data['Close'].iloc[-1], 0)
    else:
        payoff = 0

    option_price = np.exp(-r * T) * payoff
    return option_price


def historical_simulation_asian_barrier_spread_option(data, strike_price_1, strike_price_2, barrier_level, r=0.01, T=1.0):
    """
    Historical simulation for pricing an Asian-Barrier Spread option.
    """
    validate_input_parameters(strike_price_1=strike_price_1, strike_price_2=strike_price_2, r=r, T=T, barrier_level=barrier_level)
    
    average_price = data['Close'].mean()
    breached = data['Close'].max() > barrier_level

    if not breached:
        payoff = np.maximum(np.minimum(average_price - strike_price_1, strike_price_2 - strike_price_1), 0)
    else:
        payoff = 0

    option_price = np.exp(-r * T) * payoff
    return option_price

def delta_hedging_asian_option(data, strike_price, sigma, r, T, option_type="call"):
    """
    Delta hedging for an Asian option.
    """
    validate_input_parameters(strike_price=strike_price, r=r, T=T, sigma=sigma)
    
    time_to_maturity = T - np.arange(len(data)) / len(data) * T
    average_prices = data['Close'].expanding().mean()
    d1 = (np.log(average_prices / strike_price) + (r + 0.5 * sigma**2) * time_to_maturity) / (sigma * np.sqrt(time_to_maturity))
    
    if option_type == "call":
        delta = norm.cdf(d1)
    else:
        delta = -norm.cdf(-d1)
    
    cash = 0
    portfolio_value = 0
    
    for i in range(1, len(data)):
        delta_change = delta[i] - delta[i-1]
        cash -= delta_change * data['Close'][i]
        portfolio_value = delta[i] * data['Close'][i] + cash * np.exp(r * time_to_maturity[i])
    
    return portfolio_value


def delta_hedging_barrier_option(data, strike_price, barrier_level, sigma, r, T, option_type="call"):
    """
    Delta hedging for a Barrier option.
    """
    validate_input_parameters(strike_price=strike_price, r=r, T=T, sigma=sigma, barrier_level=barrier_level)
    
    time_to_maturity = T - np.arange(len(data)) / len(data) * T
    d1 = (np.log(data['Close'] / strike_price) + (r + 0.5 * sigma**2) * time_to_maturity) / (sigma * np.sqrt(time_to_maturity))
    
    if option_type == "call":
        delta = norm.cdf(d1)
    else:
        delta = -norm.cdf(-d1)
    
    cash = 0
    portfolio_value = 0
    breached = False
    
    for i in range(1, len(data)):
        if data['Close'][i] > barrier_level:
            breached = True
            break
        
        delta_change = delta[i] - delta[i-1]
        cash -= delta_change * data['Close'][i]
        portfolio_value = delta[i] * data['Close'][i] + cash * np.exp(r * time_to_maturity[i])
    
    if breached:
        return 0  # Option is knocked out
    else:
        return portfolio_value


def delta_hedging_asian_barrier_spread_option(data, strike_price_1, strike_price_2, barrier_level, sigma, r, T):
    """
    Delta hedging for an Asian-Barrier Spread option.
    """
    validate_input_parameters(strike_price_1=strike_price_1, strike_price_2=strike_price_2, r=r, T=T, sigma=sigma, barrier_level=barrier_level)
    
    time_to_maturity = T - np.arange(len(data)) / len(data) * T
    average_prices = data['Close'].expanding().mean()
    d1 = (np.log(average_prices / strike_price_1) + (r + 0.5 * sigma**2) * time_to_maturity) / (sigma * np.sqrt(time_to_maturity))
    
    delta = norm.cdf(d1)
    
    cash = 0
    portfolio_value = 0
    breached = False
    
    for i in range(1, len(data)):
        if data['Close'][i] > barrier_level:
            breached = True
            break
        
        delta_change = delta[i] - delta[i-1]
        cash -= delta_change * data['Close'][i]
        portfolio_value = delta[i] * data['Close'][i] + cash * np.exp(r * time_to_maturity[i])
    
    if breached:
        return 0  # Option is knocked out
    else:
        return portfolio_value

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from ipywidgets import interact

# Parameters
S = 100  # Current stock price
X = 105  # Strike price
T = 1    # Time to maturity (in years)
r = 0.05 # Risk-free rate
sigma = 0.2  # Volatility
N = 50  # Number of time steps for the Binomial model
n_simulations = 100000  # Number of simulations for the Monte Carlo model

# 1. Black-Scholes Model
def black_scholes_option(S, X, T, r, sigma, option_type="call"):
    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = X * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# 2. Binomial Option Pricing Model
def binomial_tree_option(S, X, T, r, sigma, N, option_type="call", american=False):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Initialize asset prices at maturity
    asset_prices = np.zeros(N + 1)
    asset_prices[0] = S * d ** N
    for i in range(1, N + 1):
        asset_prices[i] = asset_prices[i - 1] * u / d

    # Initialize option values at maturity
    option_values = np.zeros(N + 1)
    for i in range(N + 1):
        if option_type == "call":
            option_values[i] = max(0, asset_prices[i] - X)
        elif option_type == "put":
            option_values[i] = max(0, X - asset_prices[i])

    # Step back through the tree
    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            option_values[i] = np.exp(-r * dt) * (p * option_values[i + 1] + (1 - p) * option_values[i])
            if american:
                if option_type == "call":
                    option_values[i] = max(option_values[i], asset_prices[i] - X)
                elif option_type == "put":
                    option_values[i] = max(option_values[i], X - asset_prices[i])

    return option_values[0]

# 3. Monte Carlo Simulation for Option Pricing
def monte_carlo_option(S, X, T, r, sigma, n_simulations, option_type="call"):
    np.random.seed(0)  # For reproducibility
    dt = T
    discount_factor = np.exp(-r * T)
    
    # Simulate end prices using geometric Brownian motion
    S_T = S * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.randn(n_simulations))
    
    # Calculate option payoff
    if option_type == "call":
        payoffs = np.maximum(0, S_T - X)
    elif option_type == "put":
        payoffs = np.maximum(0, X - S_T)
    
    option_price = discount_factor * np.mean(payoffs)
    return option_price

# 4. Option Greeks Calculation (Black-Scholes)
def black_scholes_greeks(S, X, T, r, sigma):
    d1 = (np.log(S / X) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * X * np.exp(-r * T) * norm.cdf(d2)
    rho = X * T * np.exp(-r * T) * norm.cdf(d2)

    return delta, gamma, vega, theta, rho

# Function to display option prices and Greeks
def display_option_pricing(S, X, T, r, sigma, N, n_simulations, option_type="call", american=False):
    bs_price = black_scholes_option(S, X, T, r, sigma, option_type)
    binomial_price = binomial_tree_option(S, X, T, r, sigma, N, option_type, american)
    monte_carlo_price = monte_carlo_option(S, X, T, r, sigma, n_simulations, option_type)
    
    print(f"Black-Scholes {option_type.capitalize()} Option Price: {bs_price:.2f}")
    print(f"Binomial Tree {option_type.capitalize()} Option Price: {binomial_price:.2f}")
    print(f"Monte Carlo {option_type.capitalize()} Option Price: {monte_carlo_price:.2f}")
    
    delta, gamma, vega, theta, rho = black_scholes_greeks(S, X, T, r, sigma)
    print("\nOption Greeks (Black-Scholes Model):")
    print(f"Delta: {delta:.4f}")
    print(f"Gamma: {gamma:.4f}")
    print(f"Vega: {vega:.4f}")
    print(f"Theta: {theta:.4f}")
    print(f"Rho: {rho:.4f}")
    
    # Plot the results
    models = ['Black-Scholes', 'Binomial Tree', 'Monte Carlo']
    prices = [bs_price, binomial_price, monte_carlo_price]
    
    plt.figure(figsize=(10, 6))
    plt.bar(models, prices, color=['blue', 'green', 'orange'])
    plt.title(f'Option Pricing ({option_type.capitalize()} Option) Using Different Models')
    plt.ylabel('Option Price')
    plt.show()
    
    # Monte Carlo distribution of end prices
    S_T = S * np.exp((r - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * np.random.randn(n_simulations))
    
    plt.figure(figsize=(10, 6))
    plt.hist(S_T, bins=50, color='purple', alpha=0.7)
    plt.title('Monte Carlo Simulation of Stock Prices at Maturity')
    plt.xlabel('Stock Price at Maturity')
    plt.ylabel('Frequency')
    plt.show()

# Interactive plot to adjust parameters
interact(display_option_pricing,
         S=(50, 150, 5),
         X=(50, 150, 5),
         T=(0.1, 2.0, 0.1),
         r=(0.01, 0.10, 0.01),
         sigma=(0.1, 0.5, 0.05),
         N=(10, 100, 10),
         n_simulations=(10000, 500000, 10000),
         option_type=["call", "put"],
         american=[False, True])

import numpy as np 
from scipy.stats import norm#normal distribution funct
import matplotlib.pyplot as plt 
from ipywidgets import interact#interactive widgets

#parameters
S = 100#current stock price
X = 105#strike price of the option
T = 1#time to maturity (in years)
r = 0.05#risk-free interest rate
sigma = 0.2#volatility of the stock
N = 50#number of time steps for the Binomial model
n_simulations = 100000#number of simulations for the Monte Carlo model

#black scholes model
def black_scholes_option(S,X,T,r,sigma,option_type="call"):
    #calculate d1 and d2 using Black Scholes formula
    d1 = (np.log(S/X)+(r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    #calculate option price based on the type
    if option_type == "call":
        price = S * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)#call option price
    elif option_type == "put":
        price = X * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)#put option price
    return price

#binomial option pricing model
def binomial_tree_option(S, X, T, r, sigma, N, option_type="call", american=False):
    dt=T / N#length of each time step
    u=np.exp(sigma * np.sqrt(dt))#up factor
    d=1 / u#down factor
    p=(np.exp(r*dt)-d)/(u-d)#risk neutral probability
    #initialize asset prices at maturity
    asset_prices=np.zeros(N + 1)
    asset_prices[0]=S * d ** N#lowest price at maturity
    for i in range(1, N + 1):
        asset_prices[i]=asset_prices[i - 1] * u / d#populate all prices

    #initialize option values at maturity
    option_values=np.zeros(N + 1)
    for i in range(N + 1):
        if option_type=="call":
            option_values[i]=max(0,asset_prices[i]-X)#call option payoff
        elif option_type=="put":
            option_values[i]=max(0,X-asset_prices[i])#put option payoff

    #step backward through the tree to calculate option value at the start
    for j in range(N - 1, -1, -1):
        for i in range(j + 1):
            option_values[i]=np.exp(-r*dt)*(p*option_values[i + 1]+(1 - p)*option_values[i])
            if american:#for american options consider early exercise
                if option_type=="call":
                    option_values[i]=max(option_values[i],asset_prices[i]-X)
                elif option_type=="put":
                    option_values[i] = max(option_values[i],X-asset_prices[i])

    return option_values[0]

#monte carlo simulation for option pricing
def monte_carlo_option(S,X,T,r,sigma,n_simulations,option_type="call"):
    np.random.seed(0)#set random seed for reproducibility
    dt = T#single time step since we're simulating to maturity
    discount_factor = np.exp(-r * T)#discount factor for present value calculation
    #simulate stock prices at maturity using geometric Brownian motion
    S_T = S*np.exp((r-0.5*sigma**2)*dt+sigma*np.sqrt(dt)*np.random.randn(n_simulations))
    
    #calculate option payoff
    if option_type=="call":
        payoffs=np.maximum(0,S_T-X)#call option payoff
    elif option_type=="put":
        payoffs=np.maximum(0,X-S_T)#put option payoff
    option_price=discount_factor*np.mean(payoffs)#discounted average payoff
    return option_price

#option greeks calculation black-scholes
def black_scholes_greeks(S,X,T,r,sigma):
    d1=(np.log(S/X)+(r+0.5*sigma**2) * T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    #calculate greeks using black scholes formula
    delta=norm.cdf(d1)#sensitivity of option price to stock price
    gamma=norm.pdf(d1) / (S * sigma * np.sqrt(T))  # Sensitivity of delta to stock price
    vega=S*norm.pdf(d1)*np.sqrt(T)#sensitivity of option price to volatility
    theta=-(S*norm.pdf(d1)*sigma)/(2*np.sqrt(T))-r*X*np.exp(-r*T)*norm.cdf(d2)#time decay
    rho=X*T*np.exp(-r*T)*norm.cdf(d2)#sensitivity of option price to interest rate
    return delta, gamma, vega, theta, rho

#function to display option prices and Greeks
def display_option_pricing(S,X,T,r,sigma,N,n_simulations,option_type="call",american=False):
    #calculate option prices using different models
    bs_price=black_scholes_option(S,X,T,r,sigma, option_type)
    binomial_price=binomial_tree_option(S,X,T,r,sigma,N,option_type,american)
    monte_carlo_price=monte_carlo_option(S,X,T,r,sigma,n_simulations,option_type)
    #print calculated prices
    print(f"Black-Scholes {option_type.capitalize()} Option Price: {bs_price:.2f}")
    print(f"Binomial Tree {option_type.capitalize()} Option Price: {binomial_price:.2f}")
    print(f"Monte Carlo {option_type.capitalize()} Option Price: {monte_carlo_price:.2f}")
    # Calculate and print Greeks
    delta, gamma, vega, theta, rho = black_scholes_greeks(S, X, T, r, sigma)
    print("\nOption Greeks (Black-Scholes Model):")
    print(f"Delta:{delta:.4f}")
    print(f"Gamma:{gamma:.4f}")
    print(f"Vega:{vega:.4f}")
    print(f"Theta:{theta:.4f}")
    print(f"Rho:{rho:.4f}")
    #lot the option prices from different models
    models=['Black-Scholes','Binomial Tree','Monte Carlo']
    prices=[bs_price,binomial_price,monte_carlo_price]
    plt.figure(figsize=(10, 6))
    plt.bar(models,prices,color=['blue','green','orange'])
    plt.title(f'Option Pricing ({option_type.capitalize()} Option) Using Different Models')
    plt.ylabel('Option Price')
    plt.show()    
    #monte carlo distribution of stock prices at maturity
    S_T = S * np.exp((r-0.5*sigma**2)*T+sigma*np.sqrt(T)*np.random.randn(n_simulations))
    plt.figure(figsize=(10,6))
    plt.hist(S_T,bins=50,color='purple',alpha=0.7)
    plt.title('Monte Carlo Simulation of Stock Prices at Maturity')
    plt.xlabel('Stock Price at Maturity')
    plt.ylabel('Frequency')
    plt.show()
#interactive plot to adjust parameters
interact(display_option_pricing,
         S=(50,150,5),#range for stock price
         X=(50,150,5),#range for strike price
         T=(0.1,2.0,0.1),#range for time to maturity
         r=(0.01,0.10,0.01),#range for risk-free rate
         sigma=(0.1,0.5,0.05),#range for volatility
         N=(10,100,10),#range for time steps in binomial model
         n_simulations=(10000,500000,10000),#range for monte carlo simulations
         option_type=["call","put"],#option type: call or put
         american=[False, True])#whether the option is american or European

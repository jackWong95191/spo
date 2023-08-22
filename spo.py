import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import plotly.express as px
plt.style.use('fivethirtyeight')

st.markdown("# Stock Portfolio Optimization App")

# Define the list of Hang Seng Index stocks
tickers_list = ['0001.HK', '0002.HK', '0003.HK', '0005.HK', '0006.HK', '0011.HK', '0012.HK', '0016.HK', '0017.HK', '0027.HK',
                '0066.HK', '0101.HK', '0175.HK', '0241.HK', '0267.HK', '0288.HK', '0291.HK', '0316.HK','0322.HK', '0386.HK',
				'0388.HK','0669.HK','0688.HK','0700.HK','0762.HK','0823.HK','0836.HK','0857.HK','0868.HK','0881.HK','0883.HK',
		        '0939.HK','0941.HK','0960.HK','0968.HK','0981.HK','0992.HK','1038.HK','1044.HK','1088.HK','1093.HK','1109.HK',
				'1113.HK','1177.HK','1209.HK','1211.HK','1299.HK','1378.HK','1398.HK','1810.HK','1876.HK','1928.HK','1929.HK',
				'1997.HK','2007.HK','2020.HK','2269.HK','2313.HK','2318.HK','2319.HK','2331.HK','2382.HK','2388.HK','2628.HK',
				'2688.HK','2899.HK','3690.HK','3692.HK','3968.HK','3988.HK','6098.HK','6618.HK','6690.HK','6862.HK','9618.HK',
	            '9633.HK','9888.HK','9961.HK','9988.HK','9999.HK']
	
# Initialize the df variable
df = pd.DataFrame()

# Use Streamlit to get user inputs    
selected_stocks = st.multiselect('Select the stocks from Hang Seng Index for your portfolio', tickers_list)
weights = st.text_input('Enter the weights for the selected stocks (comma separated)').split(',')
weights = [float(i) for i in weights]
start_date = st.date_input('Select the start date', datetime(2016, 1, 1))
end_date = st.date_input('Select the end date', datetime.today())

if st.button('Calculate'):
    if len(selected_stocks) < 2 or len(selected_stocks) > 30:
        st.write('Warning: Please select between 2 to 30 stocks.')
    elif len(weights) != len(selected_stocks):
        st.write('Warning: Number of weights entered does not match the number of selected stocks.')
    elif not np.isclose(sum(weights), 1, atol=1e-8):  # tolerance for floating point precision
        st.write('Warning: Weights do not sum to 1.')
    else:
        # Download stock data
        df = yf.download(selected_stocks, start=start_date, end=end_date)['Adj Close']
		
        st.markdown("## 1. Stock Portfolio based on your assigned weighting of selected stocks")
        # show df
        st.write('The details of stock data:')
        st.dataframe(df)




# Visually show the stock/ portfolio
title = 'Portfolio Adj. Close Price History'

# Get the stocks
my_stocks = df

# Create and plot the graph
plt.figure(figsize=(20, 8))
for c in my_stocks.columns.values:
  plt.plot(my_stocks[c],label=c)

plt.title(title)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Adj. Price HKD ($)',fontsize=18)
plt.legend(my_stocks.columns.values,loc='upper left')

# Show the plot in Streamlit
st.pyplot(plt)


# show the daily simple return
st.write('The daily simple return of stock data:')
returns = df.pct_change()
st.dataframe(returns)

# Create and show the annualized covariance matrix
st.write('The annualized covariance of stock data:')
cov_matrix_annual = returns.cov()*252
st.dataframe(cov_matrix_annual)

corr_df = cov_matrix_annual.corr().round(2) # round to 2 decimal places
fig_corr = px.imshow(corr_df, text_auto=True, title = 'Correlation between Stocks')
st.plotly_chart(fig_corr)

# Convert weights to numpy array
weights = np.array(weights)
weights = weights.reshape(-1, 1)  # reshape to column vector

# Calculate the portfolio variance
port_variance = np.dot(weights.T, np.dot(cov_matrix_annual, weights))
port_variance = port_variance.item()
if len(weights.shape) == 1:
    weights = np.expand_dims(weights, axis=1)
if weights.shape[0] != cov_matrix_annual.shape[0]:
    st.write("The matrices are not compatible for multiplication.")


# Calculate the portfolio volatility aka standard deviation
port_volatility = np.sqrt(port_variance).item()

# Calculate the annual portfolio return
returns_mean = np.array(returns.mean()).flatten()
weights = np.array(weights).flatten()
portfolio_simple_annual_return = np.sum(returns_mean * weights) * 252

# Show the expected annual return, volatility (risk), and variance
percent_var = str(round(port_variance,4)*100)+'%'
percent_vola = str(round(port_volatility,4)*100)+'%'
percent_ret = str(round(portfolio_simple_annual_return,4)*100)+'%'
risk_free_rate_1 = 0.01
sharp_portfolio =  str(round((portfolio_simple_annual_return - risk_free_rate_1)/ port_volatility, 4))

st.write('Expected annual return of portfolio: '+percent_ret)
st.write('Annual volatility/ risk of portfolio: '+percent_vola)
st.write('Annual variance of portfolio: '+percent_var)
st.write('Sharpe Ratio of portfolio: ' +sharp_portfolio)

st.markdown("## 2. Finding the Optimal Portfolio of selected stocks by maximizing Sharpe Ratio")

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt import plotting

mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)
st.write('The mean of historical return of selected stocks are:')
st.write(mu)
st.write('The covaraince matrix of selected stocks are:')
st.write(S)

def plot_efficient_frontier_and_max_sharpe(mu, S):  
    # Optimize portfolio for maximal Sharpe ratio 
    ef = EfficientFrontier(mu, S)
    fig, ax = plt.subplots(figsize=(8,6))
    ef_max_sharpe = copy.deepcopy(ef)
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=False)

    # Find the max sharpe portfolio
    ef_max_sharpe.max_sharpe(risk_free_rate=0.01)
    ret_tangent, std_tangent, _ = ef_max_sharpe.portfolio_performance()
    ax.scatter(std_tangent, ret_tangent, marker="*", s=100, c="r", label="Max Sharpe")

    # Generate random portfolios
    n_samples = 1000
    w = np.random.dirichlet(np.ones(ef.n_assets), n_samples)
    rets = w.dot(ef.expected_returns)
    stds = np.sqrt(np.diag(w @ ef.cov_matrix @ w.T))
    sharpes = rets / stds
    ax.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

    # Output
    ax.set_title("Efficient Frontier with Random Portfolios")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

plot_efficient_frontier_and_max_sharpe(mu, S)

ef = EfficientFrontier(mu, S)
ef.max_sharpe(risk_free_rate=0.01)
weights = ef.clean_weights()
st.write('The optimal weights of chosen stocks are:')
st.write(weights)
 

# Display the portfolio performance
expected_return, expected_volatility, sharpe_ratio = ef.portfolio_performance()
st.write(f'Expected annual return of optimal portfolio: {expected_return*100:.4f}%')
st.write(f'Annual volatility of optimal portfolio: {expected_volatility*100:.4f}%')
st.write(f'Sharpe Ratio of optimal portfolio: {sharpe_ratio:.4f}')



st.markdown("## 3. Finding the Optimal Portfolio of selected stocks by minimizing volatility")

# Create an Efficient Frontier object for minimum volatility
ef_min_vol = EfficientFrontier(mu, S)

# Calculate the weights for the minimum volatility portfolio
raw_weights_min_vol = ef_min_vol.min_volatility()
cleaned_weights_min_vol = ef_min_vol.clean_weights()

# Get portfolio performance
expected_return_min_vol, expected_volatility_min_vol, sharpe_ratio_min_vol = ef_min_vol.portfolio_performance()

# Plot Efficient Frontier with minimum volatility portfolio
fig2, ax2 = plt.subplots(figsize=(8,6))

# Create a new Efficient Frontier object for plotting
ef_plot = EfficientFrontier(mu, S)

plotting.plot_efficient_frontier(ef_plot, ax=ax2, show_assets=False)

# Plot minimum volatility portfolio
ax2.scatter(expected_volatility_min_vol, expected_return_min_vol, marker="*", s=100, c="r", label="Min Volatility")

# Generate random portfolios
n_samples = 1000
w = np.random.dirichlet(np.ones(ef_plot.n_assets), n_samples)
rets = w.dot(ef_plot.expected_returns)
stds = np.sqrt(np.diag(w @ ef_plot.cov_matrix @ w.T))
sharpes = rets / stds
ax2.scatter(stds, rets, marker=".", c=sharpes, cmap="viridis_r")

# Output
ax2.set_title("Efficient Frontier with Random Portfolios")
ax2.legend()
plt.tight_layout()
st.pyplot(fig2)

# Display the weights for the minimum volatility portfolio
st.write('The weights for the minimum volatility portfolio are:')
st.write(cleaned_weights_min_vol)

# Calculate Sharpe ratio with risk-free rate
risk_free_rate = 0.01
sharpe_ratio_min_vol = (expected_return_min_vol - risk_free_rate) / expected_volatility_min_vol


# Display the portfolio performance
st.write(f'Expected annual return of minimum volatility portfolio: {expected_return_min_vol*100:.4f}%')
st.write(f'Annual volatility of minimum volatility portfolio: {expected_volatility_min_vol*100:.4f}%')
st.write(f'Sharpe Ratio of minimum volatility portfolio: {sharpe_ratio_min_vol:.4f}')

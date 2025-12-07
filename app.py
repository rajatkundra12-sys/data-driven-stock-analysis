import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

st.set_page_config(page_title="Nifty 50 Stock Analysis", layout="wide")

st.title("📊 Nifty 50 Stock Analysis Dashboard")
st.markdown("**Analysis Period: October 2023 - November 2024**")

symbol_files = [f for f in os.listdir('symbol_csv_files') if f.endswith('.csv')]

@st.cache_data
def load_and_calculate_returns():
    yearly_returns = {}
    for symbol_file in symbol_files:
        symbol = symbol_file.replace('.csv', '')
        try:
            df = pd.read_csv(f'symbol_csv_files/{symbol_file}')
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
            if len(df) > 1 and 'close' in df.columns:
                start_price = df['close'].iloc[0]
                end_price = df['close'].iloc[-1]
                yearly_return = (end_price - start_price) / start_price
                yearly_returns[symbol] = yearly_return
        except:
            yearly_returns[symbol] = 0
    return yearly_returns

@st.cache_data
def calculate_volatility(symbol_file):
    df = pd.read_csv(f'symbol_csv_files/{symbol_file}')
    df = df.sort_values('date')
    df['daily_return'] = df['close'].pct_change()
    volatility = df['daily_return'].std()
    return volatility

@st.cache_data
def load_sector_data():
    sector_df = pd.read_csv('Sector_data - Sheet1.csv')
    sector_df['Symbol_Clean'] = sector_df['Symbol'].str.split(':').str[1].str.strip()
    return sector_df

@st.cache_data
def calculate_monthly_returns():
    all_data = pd.DataFrame()
    
    for symbol_file in symbol_files:
        symbol = symbol_file.replace('.csv', '')
        df = pd.read_csv(f'symbol_csv_files/{symbol_file}')
        df['symbol'] = symbol
        all_data = pd.concat([all_data, df])
    
    all_data['date'] = pd.to_datetime(all_data['date'])
    all_data['month'] = all_data['date'].dt.month
    all_data['year'] = all_data['date'].dt.year
    
    all_data = all_data.sort_values(['symbol', 'date'])
    
    monthly_returns = []
    
    for symbol in all_data['symbol'].unique():
        symbol_data = all_data[all_data['symbol'] == symbol].copy()
        symbol_data = symbol_data.sort_values('date')
        
        for year in symbol_data['year'].unique():
            year_data = symbol_data[symbol_data['year'] == year]
            for month in year_data['month'].unique():
                month_data = year_data[year_data['month'] == month]
                if len(month_data) > 1:
                    start_price = month_data['close'].iloc[0]
                    end_price = month_data['close'].iloc[-1]
                    monthly_return = (end_price - start_price) / start_price
                    
                    month_name = f"{month}/{year}"
                    monthly_returns.append({
                        'symbol': symbol,
                        'month': month_name,
                        'monthly_return': monthly_return
                    })
    
    return pd.DataFrame(monthly_returns)

yearly_returns = load_and_calculate_returns()
sector_df = load_sector_data()
monthly_returns_df = calculate_monthly_returns()

returns_df = pd.DataFrame(list(yearly_returns.items()), columns=['Symbol', 'Yearly_Return'])
top_10_green = returns_df.nlargest(10, 'Yearly_Return')
top_10_loss = returns_df.nsmallest(10, 'Yearly_Return')

green_stocks = len(returns_df[returns_df['Yearly_Return'] > 0])
red_stocks = len(returns_df[returns_df['Yearly_Return'] <= 0])

all_prices = []
all_volumes = []
for symbol_file in symbol_files:
    df = pd.read_csv(f'symbol_csv_files/{symbol_file}')
    if 'close' in df.columns:
        all_prices.extend(df['close'].dropna().tolist())
    if 'volume' in df.columns:
        all_volumes.extend(df['volume'].dropna().tolist())

avg_price = np.mean(all_prices) if all_prices else 0
avg_volume = np.mean(all_volumes) if all_volumes else 0

col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Stocks", len(symbol_files))
with col2:
    st.metric("Green Stocks", green_stocks)
with col3:
    st.metric("Red Stocks", red_stocks)
with col4:
    st.metric("Average Price", f"₹{avg_price:.2f}")
with col5:
    st.metric("Average Volume", f"{avg_volume:,.0f}")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Performance", "Volatility", "Sector Analysis", "Correlation", "Monthly Analysis", "Cumulative Returns"])

with tab1:
    st.subheader("🏆 Top 10 Performing Stocks")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    top_10_green_plot = top_10_green.copy()
    top_10_green_plot['Return (%)'] = top_10_green_plot['Yearly_Return'] * 100
    
    ax1.barh(top_10_green_plot['Symbol'], top_10_green_plot['Return (%)'], color='green')
    ax1.set_xlabel('Return (%)')
    ax1.set_title('Top 10 Gainers')
    
    top_10_loss_plot = top_10_loss.copy()
    top_10_loss_plot['Return (%)'] = top_10_loss_plot['Yearly_Return'] * 100
    
    ax2.barh(top_10_loss_plot['Symbol'], top_10_loss_plot['Return (%)'], color='red')
    ax2.set_xlabel('Return (%)')
    ax2.set_title('Top 10 Losers')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.subheader("Performance Table")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Top 10 Gainers**")
        top_10_green_display = top_10_green.copy()
        top_10_green_display['Return (%)'] = (top_10_green_display['Yearly_Return'] * 100).round(2)
        st.dataframe(top_10_green_display[['Symbol', 'Return (%)']].reset_index(drop=True))
    
    with col2:
        st.write("**Top 10 Losers**")
        top_10_loss_display = top_10_loss.copy()
        top_10_loss_display['Return (%)'] = (top_10_loss_display['Yearly_Return'] * 100).round(2)
        st.dataframe(top_10_loss_display[['Symbol', 'Return (%)']].reset_index(drop=True))

with tab2:
    st.subheader("📊 Volatility Analysis")
    
    volatility_data = {}
    for symbol_file in symbol_files[:15]:
        symbol = symbol_file.replace('.csv', '')
        volatility = calculate_volatility(symbol_file)
        volatility_data[symbol] = volatility
    
    top_10_volatile = sorted(volatility_data.items(), key=lambda x: x[1], reverse=True)[:10]
    volatility_df = pd.DataFrame(top_10_volatile, columns=['Symbol', 'Volatility'])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(volatility_df['Symbol'], volatility_df['Volatility'])
    ax.set_title('Top 10 Most Volatile Stocks')
    ax.set_xlabel('Stock Symbol')
    ax.set_ylabel('Volatility')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.dataframe(volatility_df.round(4))

with tab3:
    st.subheader("🏭 Sector-wise Performance")
    
    symbol_to_sector = dict(zip(sector_df['Symbol_Clean'], sector_df['sector']))
    sector_returns = {}
    
    for symbol, return_pct in yearly_returns.items():
        sector = symbol_to_sector.get(symbol)
        if sector:
            if sector not in sector_returns:
                sector_returns[sector] = []
            sector_returns[sector].append(return_pct)
    
    sector_avg_returns = {}
    for sector, returns in sector_returns.items():
        if returns:
            sector_avg_returns[sector] = np.mean(returns)
    
    sector_performance = pd.DataFrame(list(sector_avg_returns.items()), 
                                    columns=['Sector', 'Avg_Return'])
    sector_performance = sector_performance.sort_values('Avg_Return', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(sector_performance['Sector'], sector_performance['Avg_Return'] * 100)
    ax.set_xlabel('Average Return (%)')
    ax.set_title('Sector-wise Average Returns')
    plt.tight_layout()
    st.pyplot(fig)
    
    sector_performance_display = sector_performance.copy()
    sector_performance_display['Avg_Return (%)'] = (sector_performance_display['Avg_Return'] * 100).round(2)
    st.dataframe(sector_performance_display[['Sector', 'Avg_Return (%)']])

with tab4:
    st.subheader("🔗 Stock Price Correlation")
    
    closing_prices = pd.DataFrame()
    for symbol_file in symbol_files[:12]:
        symbol = symbol_file.replace('.csv', '')
        df = pd.read_csv(f'symbol_csv_files/{symbol_file}')
        df = df.sort_values('date')
        closing_prices[symbol] = df['close']
    
    correlation_matrix = closing_prices.corr()
    
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt='.2f', linewidths=0.5, ax=ax)
    ax.set_title('Stock Price Correlation Heatmap')
    st.pyplot(fig)

with tab5:
    st.subheader("📅 Monthly Performance Analysis")
    
    month_order = ['10/2023', '11/2023', '12/2023', '1/2024', '2/2024', '3/2024',
                  '4/2024', '5/2024', '6/2024', '7/2024', '8/2024', '9/2024', '10/2024', '11/2024']
    
    available_months = [month for month in month_order if month in monthly_returns_df['month'].unique()]
    
    selected_month = st.selectbox("Select Month", available_months)
    
    if selected_month:
        month_data = monthly_returns_df[monthly_returns_df['month'] == selected_month]
        
        top_5_gainers = month_data.nlargest(5, 'monthly_return')
        top_5_losers = month_data.nsmallest(5, 'monthly_return')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.barh(top_5_gainers['symbol'], top_5_gainers['monthly_return'] * 100, color='green')
        ax1.set_title(f'Top 5 Gainers - {selected_month}')
        ax1.set_xlabel('Return (%)')
        
        ax2.barh(top_5_losers['symbol'], top_5_losers['monthly_return'] * 100, color='red')
        ax2.set_title(f'Top 5 Losers - {selected_month}')
        ax2.set_xlabel('Return (%)')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Top 5 Gainers - {selected_month}**")
            gainers_display = top_5_gainers.copy()
            gainers_display['Return (%)'] = (gainers_display['monthly_return'] * 100).round(2)
            st.dataframe(gainers_display[['symbol', 'Return (%)']].reset_index(drop=True))
        
        with col2:
            st.write(f"**Top 5 Losers - {selected_month}**")
            losers_display = top_5_losers.copy()
            losers_display['Return (%)'] = (losers_display['monthly_return'] * 100).round(2)
            st.dataframe(losers_display[['symbol', 'Return (%)']].reset_index(drop=True))

with tab6:
    st.subheader("📈 Cumulative Returns - Top 5 Performing Stocks")
    
    top_5_symbols = top_10_green['Symbol'].head(5).tolist()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for symbol in top_5_symbols:
        df = pd.read_csv(f'symbol_csv_files/{symbol}.csv')
        df = df.sort_values('date')
        df['daily_return'] = df['close'].pct_change()
        df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
        
        ax.plot(pd.to_datetime(df['date']), df['cumulative_return'] * 100, 
                label=symbol, linewidth=2)
    
    ax.set_title('Cumulative Returns - Top 5 Performing Stocks')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

st.sidebar.header("Filters")
selected_stock = st.sidebar.selectbox("Select Stock for Detailed View", returns_df['Symbol'].tolist())

if selected_stock:
    st.sidebar.subheader(f"Details for {selected_stock}")
    try:
        stock_df = pd.read_csv(f'symbol_csv_files/{selected_stock}.csv')
        stock_df['date'] = pd.to_datetime(stock_df['date'])
        
        latest_data = stock_df.iloc[-1]
        st.sidebar.write(f"Latest Close: ₹{latest_data['close']:.2f}")
        st.sidebar.write(f"Yearly Return: {yearly_returns.get(selected_stock, 0)*100:.2f}%")
        
        sector = symbol_to_sector.get(selected_stock, 'N/A')
        st.sidebar.write(f"Sector: {sector}")
        
    except:
        st.sidebar.write("Data not available")

st.markdown("---")
st.markdown("**Data Source: Nifty 50 Stocks | Period: Oct 2023 - Nov 2024**")
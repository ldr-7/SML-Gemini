import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Securities Market Line Analysis",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Bloomberg Terminal Style
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background-color: #0e1117;
    }
    .metric-container {
        background-color: rgba(20, 25, 35, 0.6);
        border: 1px solid #2a2f3a;
        border-radius: 4px;
        padding: 15px;
        margin: 5px;
    }
    .metric-label {
        color: #A0A0A0;
        font-family: 'Inter', 'Helvetica', sans-serif;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        color: #FAFAFA;
        font-family: 'Courier New', 'Roboto Mono', monospace;
        font-size: 20px;
        font-weight: 600;
        margin-top: 5px;
    }
    h1 {
        color: #FAFAFA;
        font-family: 'Inter', 'Helvetica', sans-serif;
        font-weight: 300;
        letter-spacing: 1px;
    }
    .stDataFrame {
        background-color: #0e1117;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def fetch_treasury_yield():
    """Fetch 10-Year Treasury Yield (^TNX)"""
    try:
        tnx = yf.Ticker("^TNX")
        hist = tnx.history(period="5d")
        if not hist.empty:
            current_yield = hist['Close'].iloc[-1]
            return current_yield / 100.0  # Convert to decimal
    except:
        pass
    return None

@st.cache_data(ttl=300)
def fetch_price_data(ticker, period):
    """Fetch historical price data for a ticker"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        if hist.empty:
            return None
        return hist['Close']
    except:
        return None

def calculate_log_returns(prices):
    """Calculate daily log returns"""
    return np.log(prices / prices.shift(1)).dropna()

def calculate_beta(asset_returns, market_returns):
    """Calculate Beta: Covariance(Asset, Market) / Variance(Market)"""
    if len(asset_returns) < 2 or len(market_returns) < 2:
        return np.nan
    covariance = np.cov(asset_returns, market_returns)[0][1]
    market_variance = np.var(market_returns)
    if market_variance == 0:
        return np.nan
    return covariance / market_variance

def annualize_return(daily_returns):
    """Annualize daily returns"""
    if len(daily_returns) == 0:
        return np.nan
    return np.mean(daily_returns) * 252

def annualize_volatility(daily_returns):
    """Annualize volatility"""
    if len(daily_returns) == 0:
        return np.nan
    return np.std(daily_returns) * np.sqrt(252)

def calculate_capm_metrics(tickers, benchmark_ticker, period, risk_free_rate):
    """Calculate CAPM metrics for all tickers"""
    results = []
    
    # Fetch benchmark data
    benchmark_prices = fetch_price_data(benchmark_ticker, period)
    if benchmark_prices is None or benchmark_prices.empty:
        st.error(f"Failed to fetch benchmark data for {benchmark_ticker}")
        return pd.DataFrame()
    
    benchmark_returns = calculate_log_returns(benchmark_prices)
    market_return = annualize_return(benchmark_returns)
    
    # Fetch and process each ticker
    for ticker in tickers:
        ticker = ticker.strip().upper()
        if not ticker:
            continue
            
        asset_prices = fetch_price_data(ticker, period)
        if asset_prices is None or asset_prices.empty:
            continue
        
        asset_returns = calculate_log_returns(asset_prices)
        
        # Align returns
        aligned_returns = pd.DataFrame({
            'asset': asset_returns,
            'market': benchmark_returns
        }).dropna()
        
        if len(aligned_returns) < 30:  # Need minimum data points
            continue
        
        asset_aligned = aligned_returns['asset'].values
        market_aligned = aligned_returns['market'].values
        
        # Calculate metrics
        beta = calculate_beta(asset_aligned, market_aligned)
        actual_return = annualize_return(asset_aligned)
        volatility = annualize_volatility(asset_aligned)
        
        if np.isnan(beta) or np.isnan(actual_return):
            continue
        
        # CAPM Expected Return
        expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
        
        # Jensen's Alpha
        alpha = actual_return - expected_return
        
        results.append({
            'Ticker': ticker,
            'Beta': beta,
            'Actual Return': actual_return,
            'Expected Return': expected_return,
            'Alpha': alpha,
            'Volatility': volatility
        })
    
    return pd.DataFrame(results), market_return

def create_sml_plot(df, market_return, risk_free_rate):
    """Create Securities Market Line scatter plot"""
    if df.empty:
        return go.Figure()
    
    # Create SML line
    beta_range = np.linspace(df['Beta'].min() - 0.2, df['Beta'].max() + 0.2, 100)
    sml_line = risk_free_rate + beta_range * (market_return - risk_free_rate)
    
    # Color points by Alpha
    colors = ['#00FFFF' if alpha > 0 else '#FF0055' for alpha in df['Alpha']]
    
    fig = go.Figure()
    
    # Add SML line
    fig.add_trace(go.Scatter(
        x=beta_range,
        y=sml_line,
        mode='lines',
        name='SML (Fair Value)',
        line=dict(color='#FAFAFA', width=2, dash='dash'),
        showlegend=True
    ))
    
    # Add scatter points
    for idx, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Beta']],
            y=[row['Actual Return']],
            mode='markers',
            name=row['Ticker'],
            marker=dict(
                size=12,
                color=colors[idx],
                line=dict(width=1, color='#0e1117')
            ),
            text=f"Ticker: {row['Ticker']}<br>" +
                 f"Beta: {row['Beta']:.4f}<br>" +
                 f"Actual Return: {row['Actual Return']:.4f}<br>" +
                 f"Expected Return: {row['Expected Return']:.4f}<br>" +
                 f"Alpha: {row['Alpha']:.4f}<br>" +
                 f"Volatility: {row['Volatility']:.4f}",
            hovertemplate='%{text}<extra></extra>',
            showlegend=False
        ))
    
    fig.update_layout(
        title={
            'text': 'Securities Market Line Analysis',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#FAFAFA', 'family': 'Inter'}
        },
        xaxis=dict(
            title='Beta (Systematic Risk)',
            title_font=dict(color='#FAFAFA', family='Inter'),
            tickfont=dict(color='#A0A0A0', family='Courier New'),
            gridcolor='#2a2f3a',
            gridwidth=1,
            zeroline=False
        ),
        yaxis=dict(
            title='Annualized Return',
            title_font=dict(color='#FAFAFA', family='Inter'),
            tickfont=dict(color='#A0A0A0', family='Courier New'),
            gridcolor='#2a2f3a',
            gridwidth=1,
            zeroline=False
        ),
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#FAFAFA'),
        height=600,
        hovermode='closest'
    )
    
    return fig

def style_dataframe(df):
    """Apply heatmap styling to Alpha column"""
    if df.empty:
        return df
    
    def color_alpha(val):
        if val > 0:
            return 'background-color: rgba(0, 255, 255, 0.2); color: #00FFFF'
        else:
            return 'background-color: rgba(255, 0, 85, 0.2); color: #FF0055'
    
    styled = df.style.applymap(color_alpha, subset=['Alpha'])
    styled = styled.format({
        'Beta': '{:.4f}',
        'Actual Return': '{:.4f}',
        'Expected Return': '{:.4f}',
        'Alpha': '{:.4f}',
        'Volatility': '{:.4f}'
    })
    
    return styled

# Sidebar
with st.sidebar:
    st.markdown("### Input Parameters")
    
    ticker_input = st.text_area(
        "Ticker List",
        value="AAPL\nMSFT\nGOOGL\nAMZN\nTSLA\nNVDA\nMETA\nNFLX",
        height=150,
        help="Enter tickers separated by newlines"
    )
    
    benchmark_ticker = st.text_input(
        "Benchmark Ticker",
        value="SPY",
        help="Market benchmark (e.g., SPY, QQQ)"
    )
    
    period_options = {
        "1Y": "1y",
        "2Y": "2y",
        "5Y": "5y",
        "YTD": "ytd"
    }
    
    period_label = st.selectbox(
        "Lookback Period",
        options=list(period_options.keys()),
        index=0
    )
    period = period_options[period_label]
    
    # Risk-free rate
    st.markdown("### Risk-Free Rate")
    auto_rf = fetch_treasury_yield()
    
    if auto_rf is not None:
        st.info(f"10-Year Treasury Yield: {auto_rf:.4f}")
        use_auto = st.checkbox("Use Auto-Fetched Rate", value=True)
        if use_auto:
            risk_free_rate = auto_rf
        else:
            risk_free_rate = st.number_input(
                "Manual Risk-Free Rate",
                min_value=0.0,
                max_value=1.0,
                value=0.02,
                step=0.001,
                format="%.4f"
            )
    else:
        st.warning("Could not fetch Treasury yield. Using manual input.")
        risk_free_rate = st.number_input(
            "Risk-Free Rate",
            min_value=0.0,
            max_value=1.0,
            value=0.02,
            step=0.001,
            format="%.4f"
        )
    
    st.markdown("---")
    calculate_button = st.button("Calculate", type="primary", use_container_width=True)

# Main Content
st.markdown("<h1>Securities Market Line Analysis</h1>", unsafe_allow_html=True)

if calculate_button or 'results' not in st.session_state:
    tickers = [t for t in ticker_input.split('\n') if t.strip()]
    
    if not tickers:
        st.warning("Please enter at least one ticker symbol.")
    else:
        with st.spinner("Fetching data and calculating metrics..."):
            df, market_return = calculate_capm_metrics(
                tickers, benchmark_ticker, period, risk_free_rate
            )
            
            if df.empty:
                st.error("No valid data retrieved. Please check ticker symbols and try again.")
            else:
                st.session_state['results'] = df
                st.session_state['market_return'] = market_return
                st.session_state['risk_free_rate'] = risk_free_rate
                st.session_state['benchmark'] = benchmark_ticker

if 'results' in st.session_state:
    df = st.session_state['results']
    market_return = st.session_state['market_return']
    risk_free_rate = st.session_state['risk_free_rate']
    benchmark = st.session_state['benchmark']
    
    # KPI Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div class="metric-container">
                <div class="metric-label">Market Return ({})</div>
                <div class="metric-value">{:.4f}</div>
            </div>
        """.format(benchmark, market_return), unsafe_allow_html=True)
    
    if not df.empty:
        best_alpha_ticker = df.loc[df['Alpha'].idxmax(), 'Ticker']
        best_alpha_value = df['Alpha'].max()
        worst_alpha_ticker = df.loc[df['Alpha'].idxmin(), 'Ticker']
        worst_alpha_value = df['Alpha'].min()
        highest_beta_ticker = df.loc[df['Beta'].idxmax(), 'Ticker']
        highest_beta_value = df['Beta'].max()
        
        with col2:
            color = '#00FFFF' if best_alpha_value > 0 else '#FF0055'
            st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Best Alpha</div>
                    <div class="metric-value" style="color: {color}">{best_alpha_ticker}: {best_alpha_value:.4f}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            color = '#00FFFF' if worst_alpha_value > 0 else '#FF0055'
            st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Worst Alpha</div>
                    <div class="metric-value" style="color: {color}">{worst_alpha_ticker}: {worst_alpha_value:.4f}</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Highest Beta</div>
                    <div class="metric-value">{highest_beta_ticker}: {highest_beta_value:.4f}</div>
                </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main Visualization
    fig = create_sml_plot(df, market_return, risk_free_rate)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Data Table
    st.markdown("### Detailed Metrics")
    styled_df = style_dataframe(df)
    st.dataframe(
        styled_df,
        use_container_width=True,
        height=400
    )

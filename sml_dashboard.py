import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Universe Presets
SECTOR_ETFS = ["XLF", "XLK", "XLE", "XLV", "XLY", "XLP", "XLI", "XLC", "XLU", "XLB", "XLRE", "SMH", "IBB", "KRE"]
DOW_30 = ["AAPL", "MSFT", "JPM", "V", "UNH", "PG", "HD", "JNJ", "MRK", "CRM", "CVX", "KO", "WMT", "CSCO", "MCD", "DIS", "CAT", "AXP", "IBM", "GS", "AMGN", "VZ", "BA", "HON", "NKE", "TRV", "MMM", "DOW", "INTC", "WBA"]
QQQ_TOP = ["AAPL", "MSFT", "NVDA", "AVGO", "AMZN", "META", "TSLA", "GOOGL", "GOOG", "COST", "ADBE", "NFLX", "AMD", "PEP", "CSCO", "INTC", "TMUS", "CMCSA", "INTU", "AMGN", "QCOM", "TXN", "HON", "AMAT", "BKNG", "SBUX", "GILD", "ISRG", "MDLZ", "ADP", "LRCX", "REGN", "VRTX", "ADI", "PANW", "MU"]
MAG_7 = ["NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]

UNIVERSE_PRESETS = {
    "Custom Input": (None, None),
    "Sector ETFs": (SECTOR_ETFS, "SPY"),
    "Dow 30": (DOW_30, "DIA"),
    "Nasdaq 100": (QQQ_TOP, "QQQ"),
    "Magnificent 7": (MAG_7, "SPY")
}

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
    asset_returns_dict = {}  # Store individual asset returns for portfolio calculation
    
    # Fetch benchmark data
    benchmark_prices = fetch_price_data(benchmark_ticker, period)
    if benchmark_prices is None or benchmark_prices.empty:
        st.error(f"Failed to fetch benchmark data for {benchmark_ticker}")
        return pd.DataFrame(), {}, pd.Series(dtype=float), np.nan
    
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
        
        # Store aligned returns for portfolio calculation
        asset_returns_dict[ticker] = aligned_returns['asset']
        
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
    
    return pd.DataFrame(results), asset_returns_dict, benchmark_returns, market_return

def calculate_portfolio_metrics(asset_returns_dict, benchmark_returns, market_return, risk_free_rate):
    """Calculate Equal-Weighted Portfolio metrics"""
    if not asset_returns_dict:
        return None
    
    # Align all returns to common dates
    returns_df = pd.DataFrame(asset_returns_dict)
    returns_df = returns_df.dropna()  # Drop rows with any NaN
    
    if returns_df.empty or len(returns_df) < 30:
        return None
    
    # Equal-weighted portfolio returns (simple average)
    portfolio_returns = returns_df.mean(axis=1)
    
    # Align portfolio returns with benchmark
    aligned = pd.DataFrame({
        'portfolio': portfolio_returns,
        'market': benchmark_returns
    }).dropna()
    
    if len(aligned) < 30:
        return None
    
    portfolio_aligned = aligned['portfolio'].values
    market_aligned = aligned['market'].values
    
    # Calculate portfolio metrics
    portfolio_beta = calculate_beta(portfolio_aligned, market_aligned)
    portfolio_return = annualize_return(portfolio_aligned)
    
    if np.isnan(portfolio_beta) or np.isnan(portfolio_return):
        return None
    
    # CAPM Expected Return
    expected_return = risk_free_rate + portfolio_beta * (market_return - risk_free_rate)
    
    # Jensen's Alpha
    portfolio_alpha = portfolio_return - expected_return
    
    return {
        'Ticker': 'PORTFOLIO',
        'Beta': portfolio_beta,
        'Actual Return': portfolio_return,
        'Expected Return': expected_return,
        'Alpha': portfolio_alpha,
        'Volatility': annualize_volatility(portfolio_aligned)
    }

def create_sml_plot(df, market_return, risk_free_rate, portfolio_metrics=None):
    """Create Securities Market Line scatter plot"""
    if df.empty:
        return go.Figure()
    
    # Create SML line
    beta_min = df['Beta'].min()
    beta_max = df['Beta'].max()
    if portfolio_metrics:
        beta_min = min(beta_min, portfolio_metrics['Beta'])
        beta_max = max(beta_max, portfolio_metrics['Beta'])
    
    beta_range = np.linspace(beta_min - 0.2, beta_max + 0.2, 100)
    sml_line = risk_free_rate + beta_range * (market_return - risk_free_rate)
    
    # Color points by Alpha
    colors = ['#00FFFF' if alpha > 0 else '#FF0055' for alpha in df['Alpha']]
    
    # Create customdata using np.stack for proper 2D array structure
    custom_data_array = np.stack((
        df['Ticker'].values,
        df['Alpha'].values,
        df['Expected Return'].values,
        df['Beta'].values,
        df['Actual Return'].values,
        df['Volatility'].values
    ), axis=-1)
    
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
    
    # Add scatter points with markers+text mode
    fig.add_trace(go.Scatter(
        x=df['Beta'].values,
        y=df['Actual Return'].values,
        mode='markers+text',
        text=df['Ticker'].values,
        textposition='top center',
        textfont=dict(size=10, color='#FAFAFA'),
        name='Assets',
        marker=dict(
            size=12,
            color=colors,
            line=dict(width=1, color='#0e1117')
        ),
        customdata=custom_data_array,
        hovertemplate='<b>Ticker: %{customdata[0]}</b><br>' +
                     'Beta: %{x:.4f}<br>' +
                     'Actual Return: %{y:.4f}<br>' +
                     'Expected Return: %{customdata[2]:.4f}<br>' +
                     'Alpha: %{customdata[1]:.4f}<br>' +
                     'Volatility: %{customdata[5]:.4f}<extra></extra>',
        showlegend=False
    ))
    
    # Add Portfolio point as large gold star
    if portfolio_metrics:
        fig.add_trace(go.Scatter(
            x=[portfolio_metrics['Beta']],
            y=[portfolio_metrics['Actual Return']],
            mode='markers+text',
            text=['PORTFOLIO'],
            textposition='top center',
            textfont=dict(size=12, color='#FFD700', family='Inter', weight='bold'),
            name='Portfolio',
            marker=dict(
                symbol='star',
                size=20,
                color='#FFD700',
                line=dict(width=2, color='#0e1117')
            ),
            customdata=[[
                portfolio_metrics['Ticker'],
                portfolio_metrics['Alpha'],
                portfolio_metrics['Expected Return'],
                portfolio_metrics['Beta'],
                portfolio_metrics['Actual Return'],
                portfolio_metrics['Volatility']
            ]],
            hovertemplate='<b>PORTFOLIO (Equal-Weighted)</b><br>' +
                         'Beta: %{x:.4f}<br>' +
                         'Actual Return: %{y:.4f}<br>' +
                         'Expected Return: %{customdata[2]:.4f}<br>' +
                         'Alpha: %{customdata[1]:.4f}<br>' +
                         'Volatility: %{customdata[5]:.4f}<extra></extra>',
            showlegend=True
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

def calculate_risk_metrics(asset_returns_dict):
    """Calculate VaR, CVaR, and Max Drawdown for each asset"""
    risk_results = []
    
    for ticker, returns in asset_returns_dict.items():
        if len(returns) < 30:
            continue
        
        returns_array = returns.values
        
        # VaR (95%) - 5th percentile
        var_95 = np.percentile(returns_array, 5)
        
        # CVaR (95%) - Expected Shortfall (average of returns below 5th percentile)
        cvar_95 = returns_array[returns_array <= var_95].mean()
        if np.isnan(cvar_95):
            cvar_95 = var_95
        
        # Max Drawdown calculation
        # Convert log returns to cumulative returns
        cumulative_returns = np.exp(np.cumsum(returns_array))
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        risk_results.append({
            'Ticker': ticker,
            'VaR (95%)': var_95,
            'CVaR (95%)': cvar_95,
            'Max Drawdown': max_drawdown
        })
    
    return pd.DataFrame(risk_results)

def style_risk_dataframe(df):
    """Apply heatmap styling to risk metrics (Red = High Risk, Green = Low Risk)"""
    if df.empty:
        return df
    
    def color_risk(val, col_name):
        """Color based on risk level"""
        if col_name == 'VaR (95%)' or col_name == 'CVaR (95%)':
            # More negative = higher risk (red), less negative = lower risk (green)
            if val < -0.05:  # Very high risk
                return 'background-color: rgba(255, 0, 0, 0.4); color: #FF0000'
            elif val < -0.02:  # High risk
                return 'background-color: rgba(255, 100, 100, 0.3); color: #FF6464'
            elif val < 0:  # Moderate risk
                return 'background-color: rgba(255, 200, 0, 0.2); color: #FFC800'
            else:  # Low risk
                return 'background-color: rgba(0, 255, 0, 0.2); color: #00FF00'
        elif col_name == 'Max Drawdown':
            # More negative = higher risk (red), less negative = lower risk (green)
            if val < -0.5:  # Very high risk
                return 'background-color: rgba(255, 0, 0, 0.4); color: #FF0000'
            elif val < -0.3:  # High risk
                return 'background-color: rgba(255, 100, 100, 0.3); color: #FF6464'
            elif val < -0.15:  # Moderate risk
                return 'background-color: rgba(255, 200, 0, 0.2); color: #FFC800'
            else:  # Low risk
                return 'background-color: rgba(0, 255, 0, 0.2); color: #00FF00'
        return ''
    
    # Apply styling column by column
    styled = df.style
    for col in ['VaR (95%)', 'CVaR (95%)', 'Max Drawdown']:
        if col in df.columns:
            styled = styled.applymap(
                lambda val: color_risk(val, col),
                subset=[col]
            )
    
    styled = styled.format({
        'VaR (95%)': '{:.4f}',
        'CVaR (95%)': '{:.4f}',
        'Max Drawdown': '{:.4f}'
    })
    
    return styled

def calculate_correlation_matrix(asset_returns_dict):
    """Calculate correlation matrix of log returns"""
    if not asset_returns_dict:
        return None
    
    # Align all returns to common dates
    returns_df = pd.DataFrame(asset_returns_dict)
    returns_df = returns_df.dropna()
    
    if returns_df.empty:
        return None
    
    # Calculate correlation matrix
    corr_matrix = returns_df.corr()
    
    return corr_matrix

def create_correlation_heatmap(corr_matrix):
    """Create Plotly heatmap for correlation matrix"""
    if corr_matrix is None or corr_matrix.empty:
        return go.Figure()
    
    fig = px.imshow(
        corr_matrix,
        color_continuous_scale='RdBu_r',
        aspect='auto',
        text_auto='.2f',
        labels=dict(color='Correlation')
    )
    
    fig.update_layout(
        title={
            'text': 'Asset Correlation Matrix',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18, 'color': '#FAFAFA', 'family': 'Inter'}
        },
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#FAFAFA'),
        height=600,
        xaxis=dict(
            tickfont=dict(color='#A0A0A0', family='Courier New'),
            title_font=dict(color='#FAFAFA', family='Inter')
        ),
        yaxis=dict(
            tickfont=dict(color='#A0A0A0', family='Courier New'),
            title_font=dict(color='#FAFAFA', family='Inter')
        ),
        coloraxis_colorbar=dict(
            tickfont=dict(color='#A0A0A0', family='Courier New'),
            title_font=dict(color='#FAFAFA', family='Inter')
        )
    )
    
    return fig

# Sidebar
with st.sidebar:
    st.markdown("### Input Parameters")
    
    # Universe Preset Selector
    universe_selection = st.selectbox(
        "Universe",
        options=list(UNIVERSE_PRESETS.keys()),
        index=0,
        help="Select a preset universe or use Custom Input",
        key='universe_selectbox'
    )
    
    # Initialize session state
    if 'ticker_input_value' not in st.session_state:
        st.session_state['ticker_input_value'] = "AAPL\nMSFT\nGOOGL\nAMZN\nTSLA\nNVDA\nMETA\nNFLX"
    if 'last_universe_selection' not in st.session_state:
        st.session_state['last_universe_selection'] = universe_selection
    if 'benchmark_value' not in st.session_state:
        st.session_state['benchmark_value'] = "SPY"
    
    # Update ticker input and benchmark when universe selection changes
    if universe_selection != st.session_state['last_universe_selection']:
        if universe_selection != "Custom Input":
            ticker_list, benchmark = UNIVERSE_PRESETS[universe_selection]
            st.session_state['ticker_input_value'] = "\n".join(ticker_list)
            st.session_state['benchmark_value'] = benchmark
        st.session_state['last_universe_selection'] = universe_selection
    
    ticker_input = st.text_area(
        "Asset List",
        value=st.session_state['ticker_input_value'],
        height=150,
        help="Enter tickers separated by newlines",
        key='ticker_input_area'
    )
    
    # Always update session state with current text area value
    st.session_state['ticker_input_value'] = ticker_input
    
    # Set benchmark based on universe selection
    if universe_selection != "Custom Input":
        _, default_benchmark = UNIVERSE_PRESETS[universe_selection]
        benchmark_ticker = st.text_input(
            "Benchmark Ticker",
            value=default_benchmark,
            help="Market benchmark (e.g., SPY, QQQ, DIA)",
            key='benchmark_input'
        )
    else:
        benchmark_ticker = st.text_input(
            "Benchmark Ticker",
            value=st.session_state.get('benchmark_value', 'SPY'),
            help="Market benchmark (e.g., SPY, QQQ, DIA)",
            key='benchmark_input'
        )
        st.session_state['benchmark_value'] = benchmark_ticker
    
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
            df, asset_returns_dict, benchmark_returns, market_return = calculate_capm_metrics(
                tickers, benchmark_ticker, period, risk_free_rate
            )
            
            if df.empty:
                st.error("No valid data retrieved. Please check ticker symbols and try again.")
            else:
                # Calculate portfolio metrics
                portfolio_metrics = calculate_portfolio_metrics(
                    asset_returns_dict, benchmark_returns, market_return, risk_free_rate
                )
                
                st.session_state['results'] = df
                st.session_state['asset_returns_dict'] = asset_returns_dict
                st.session_state['benchmark_returns'] = benchmark_returns
                st.session_state['market_return'] = market_return
                st.session_state['risk_free_rate'] = risk_free_rate
                st.session_state['benchmark'] = benchmark_ticker
                st.session_state['portfolio_metrics'] = portfolio_metrics

if 'results' in st.session_state:
    df = st.session_state['results']
    market_return = st.session_state['market_return']
    risk_free_rate = st.session_state['risk_free_rate']
    benchmark = st.session_state['benchmark']
    portfolio_metrics = st.session_state.get('portfolio_metrics', None)
    asset_returns_dict = st.session_state.get('asset_returns_dict', {})
    benchmark_returns = st.session_state.get('benchmark_returns', None)
    
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
    fig = create_sml_plot(df, market_return, risk_free_rate, portfolio_metrics)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Advanced Risk Tabs
    tab1, tab2, tab3 = st.tabs(["Alpha Metrics", "Risk Desk", "Correlations"])
    
    with tab1:
        st.markdown("### Alpha Metrics")
        
        # Include portfolio in the display dataframe if available
        display_df = df.copy()
        if portfolio_metrics:
            portfolio_row = pd.DataFrame([portfolio_metrics])
            display_df = pd.concat([display_df, portfolio_row], ignore_index=True)
        
        styled_df = style_dataframe(display_df)
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=400
        )
        
        # CSV Download Button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download Alpha Metrics as CSV",
            data=csv,
            file_name=f"alpha_metrics_{benchmark}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with tab2:
        st.markdown("### Risk Desk")
        
        if asset_returns_dict:
            risk_df = calculate_risk_metrics(asset_returns_dict)
            
            if not risk_df.empty:
                styled_risk_df = style_risk_dataframe(risk_df)
                st.dataframe(
                    styled_risk_df,
                    use_container_width=True,
                    height=400
                )
                
                st.markdown("""
                    <div style="margin-top: 15px; padding: 10px; background-color: rgba(20, 25, 35, 0.6); border-radius: 4px;">
                        <p style="color: #A0A0A0; font-size: 12px; margin: 0;">
                            <b>Risk Metrics Definitions:</b><br>
                            <b>VaR (95%):</b> Historical Value at Risk - 5th percentile of returns (worst 5% of days)<br>
                            <b>CVaR (95%):</b> Conditional VaR (Expected Shortfall) - Average return on worst 5% of days<br>
                            <b>Max Drawdown:</b> Maximum observed loss from peak to trough over the period
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Insufficient data to calculate risk metrics.")
        else:
            st.warning("No asset returns data available.")
    
    with tab3:
        st.markdown("### Correlations")
        
        if asset_returns_dict:
            corr_matrix = calculate_correlation_matrix(asset_returns_dict)
            
            if corr_matrix is not None and not corr_matrix.empty:
                corr_fig = create_correlation_heatmap(corr_matrix)
                st.plotly_chart(corr_fig, use_container_width=True)
            else:
                st.warning("Insufficient data to calculate correlation matrix.")
        else:
            st.warning("No asset returns data available.")

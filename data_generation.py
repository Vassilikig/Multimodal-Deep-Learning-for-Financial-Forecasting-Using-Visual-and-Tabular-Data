import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import mplfinance as mpf
from tqdm.notebook import tqdm
import os
os.environ['TQDM_DISABLE'] = '0' 
os.environ['TQDM_NOTEBOOK'] = '0' 
import ipywidgets
import fredapi as fred
import pandas_datareader.data as web
import yfinance as yf
from dateutil.relativedelta import relativedelta
import seaborn as sns
from datetime import datetime

def get_stock_data(ticker, start_date, end_date):
    """
    Download stock data from yfinance

    Parameters:
    ticker : str
        Stock ticker
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format

    Returns:
    df : DataFrame
        Restructured DataFrame compatible with mplfinance
    """
    data = yf.download(ticker, start=start_date, end=end_date, progress=False)

    # Check whether data comes in MultiIndex format and flatten if that is the case.
    if isinstance(data.columns, pd.MultiIndex):
        new_data = pd.DataFrame(index=data.index)

        if len(data.columns) > 0:
            # Get the first column to check structure
            first_col = data.columns[0]
            
            
            # Define OHLCV columns we want to extract 
            ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
            
            # Try both possible MultiIndex orientations with case-insensitive matching
            if first_col[0].lower() in ohlcv_cols:
                for col in ohlcv_cols:
                    # Find matching column
                    matching_cols = [c for c in data.columns if c[0].lower() == col and c[1] == ticker]
                    if matching_cols:
                        new_data[col.capitalize()] = data[matching_cols[0]]
            else:
                for col in ohlcv_cols:
                    # Find matching column 
                    matching_cols = [c for c in data.columns if c[1].lower() == col and c[0] == ticker]
                    if matching_cols:
                        new_data[col.capitalize()] = data[matching_cols[0]]

        # Convert only OHLCV columns to float
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col in new_data.columns:
                new_data[col] = new_data[col].astype(float)

        # Check if we have all required columns
        missing = [col for col in required_cols if col not in new_data.columns]
        if missing:
            raise ValueError(f"Missing required columns after restructuring: {missing}")

        return new_data
        
    else:
        # Only convert OHLCV columns to float
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in ohlcv_cols:
            if col in data.columns:
                data[col] = data[col].astype(float)
        return data

def generate_candlestick_chart(data, ticker, time_window, output_file, lookback_days=14):
    """
    Generate 14-day candlestick chart with 1344×1344 pixel size,

    Parameters:
    data : DataFrame
        Stock data with OHLCV columns
    ticker : str
        Stock ticker
    time_window : tuple
        (start_date, end_date) for the window
    output_file : str
        Output filename
    lookback_days : int
        Number of days to include in the chart

    Returns:
    bool
        True if successful, False otherwise
    """
    window_start, window_end = time_window

    try:
        # Convert dates to pandas datetime 
        if isinstance(window_start, str):
            window_start = pd.to_datetime(window_start)
        if isinstance(window_end, str):
            window_end = pd.to_datetime(window_end)

        # Calculate start date for a lookback_days window ending at window_end
        chart_start = window_end - pd.Timedelta(days=lookback_days)

        # Get data for this time window
        window_data = data[data.index >= chart_start]
        window_data = window_data[window_data.index <= window_end]

        # Check for enough data
        if len(window_data) < 5:
            print(f"Not enough data for {ticker} in {lookback_days}-day window ending {window_end}")
            
            return False

        # Enhanced visual quality for visual transformer
        mc = mpf.make_marketcolors(
            up='#00873c', 
            down='#d32f2f',
            edge={'up':'#00873c', 'down':'#d32f2f'},
            wick={'up':'#00873c', 'down':'#d32f2f'},
            ohlc={'up':'#00873c', 'down':'#d32f2f'}
        )

        s = mpf.make_mpf_style(
            marketcolors=mc,
            figcolor='white',
            gridstyle='--',
            gridcolor='#e0e0e0',
            edgecolor='white',
            y_on_right=False,
            rc={'lines.linewidth': 1.5}
        )

        # Larger initial figure size and resize later for better quality
        figsize_inches = (4.48, 4.48)

        # Create the plot
        fig, axes = mpf.plot(
            window_data,
            type='candle',
            style=s,
            volume=False, # Remove volume data from the charts
            figsize=figsize_inches,
            returnfig=True,
            tight_layout=False)

        # Remove all axis elements as we want clean image for the vision transformer
        for ax in axes:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title('')

            for spine in ax.spines.values():
                spine.set_visible(False)

        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0, hspace=0)

        # Save
        plt.savefig(
            output_file,
            dpi=300,
            format='png',
            facecolor='white',
            edgecolor='none',
            bbox_inches='tight',
            pad_inches=0.01   # Small padding to prevent cutting off candlesticks
        )

        plt.close(fig) 

        return True

    except Exception as e:
        print(f"Error generating chart for {ticker} in window {window_start} to {window_end}: {str(e)}")
        
        return False

def calculate_technical_indicators(data):
    """
    Calculate technical indicators for a given stock dataframe
    Parameters:
    -----------
    data : DataFrame
        Stock data with OHLCV columns
    Returns:
    --------
    DataFrame
        Original data with added technical indicators
    """
    # Create copy to avoid modifying the original data
    df = data.copy()
    
    # first, calculate all future returns - these will be our target variables 
    close_future_1d = df['Close'].shift(-1)
    df['Return_1d'] = (close_future_1d / df['Close'] - 1) * 100
    
    close_future_5d = df['Close'].shift(-5)
    df['Return_5d'] = (close_future_5d / df['Close'] - 1) * 100
    
    close_future_10d = df['Close'].shift(-10)
    df['Return_10d'] = (close_future_10d / df['Close'] - 1) * 100
    
    # add lagged returns (past returns as features)
    df['Lagged_1d'] = df['Close'].pct_change(periods=1) * 100
    df['Lagged_2d'] = df['Close'].pct_change(periods=2) * 100
    df['Lagged_3d'] = df['Close'].pct_change(periods=3) * 100
    df['Lagged_4d'] = df['Close'].pct_change(periods=4) * 100
    df['Lagged_5d'] = df['Close'].pct_change(periods=5) * 100
    df['Lagged_6d'] = df['Close'].pct_change(periods=6) * 100
    df['Lagged_7d'] = df['Close'].pct_change(periods=7) * 100
    df['Lagged_8d'] = df['Close'].pct_change(periods=8) * 100
    df['Lagged_9d'] = df['Close'].pct_change(periods=9) * 100
    df['Lagged_10d'] = df['Close'].pct_change(periods=10) * 100
    df['Lagged_11d'] = df['Close'].pct_change(periods=11) * 100
    df['Lagged_12d'] = df['Close'].pct_change(periods=12) * 100
    df['Lagged_13d'] = df['Close'].pct_change(periods=13) * 100
    df['Lagged_14d'] = df['Close'].pct_change(periods=14) * 100
        
    # Now calculate technical indicators
    df['SMA_7'] = df['Close'].rolling(window=7).mean()
    df['SMA_14'] = df['Close'].rolling(window=14).mean()
    df['SMA_21'] = df['Close'].rolling(window=21).mean()
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    # Avoid division by zero
    loss = loss.replace(0, np.nan)
    rs = gain / loss
    rs = rs.fillna(0)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=14).mean()
    df['BB_StdDev'] = df['Close'].rolling(window=14).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_StdDev'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_StdDev'] * 2)
    
    # Daily Returns and Volatility
    df['Daily_Return'] = df['Close'].pct_change() * 100
    df['Volatility_20d'] = df['Daily_Return'].rolling(window=20).std()
    
    # Only fill NaN values for features (historical indicators) with forward fill
    target_cols = ['Return_1d', 'Return_5d', 'Return_10d']
    feature_cols = [col for col in df.columns if col not in target_cols]
    
    lagged_cols = ['Lagged_1d', 'Lagged_2d', 'Lagged_3d', 'Lagged_4d', 'Lagged_5d', 'Lagged_6d', 'Lagged_7d', 'Lagged_8d',
                  'Lagged_9d', 'Lagged_10d', 'Lagged_11d', 'Lagged_12d', 'Lagged_13d', 'Lagged_14d']
    for col in lagged_cols:
        df[col] = df[col].fillna(0)
    
    # Forward fill other features
    df[feature_cols] = df[feature_cols].ffill()

    df = df.dropna(subset=['Return_1d', 'Return_5d', 'Return_10d'])
    
    return df

def create_time_windows(data, window_size_days=14, step_size_days=7):
    """
    Create time windows ensuring each window has valid return data for training
    """
    # First make sure Return_1d is in the data
    if 'Return_1d' not in data.columns:
        print("Warning: Return_1d column not found in data. Cannot create windows.")
        return []
        
    # Ensure index is datetime
    if not isinstance(data.index, pd.DatetimeIndex):
        data.index = pd.to_datetime(data.index)
        
    # First get all dates in the dataset
    all_dates = data.index.sort_values()
    windows = []
    window_count = 0
    skipped_count = 0
    
    # We need to iterate through potential window start dates
    # Use range with step to correctly increment i
    for i in range(0, len(all_dates) - window_size_days, step_size_days):
        # Define window start and end
        window_start = all_dates[i]
        window_end = all_dates[min(i + window_size_days, len(all_dates) - 1)]  # Add bounds check
        
        # Get data for the window
        window_data = data[(data.index >= window_start) & (data.index <= window_end)]
        
        # Check if we're near the end of our dataset - we need next-day data for Returns
        has_next_day = i + window_size_days < len(all_dates) - 1
        
        # Only proceed if we have at least one data point and have next-day data
        if len(window_data) > 0 and has_next_day:
            # Check if last day has valid return data
            last_day_return = window_data['Return_1d'].iloc[-1]
            has_valid_return = not pd.isna(last_day_return)
            
            if has_valid_return:
                windows.append((window_start, window_end))
                window_count += 1
            else:
                skipped_count += 1
        else:
            skipped_count += 1
    
    print(f"Created {window_count} valid windows, skipped {skipped_count} invalid windows")
    
    return windows

MACRO_INDICATORS = {
    # Interest Rates/ Yield Curve
    'DGS10': '10-Year Treasury Rate',
    'DGS2': '2-Year Treasury Rate',
    'T10Y2Y': '10-Year Treasury Minus 2-Year Treasury (Yield Curve)',
    'FEDFUNDS': 'Federal Funds Effective Rate',

    # Vol and Risk
    'VIXCLS': 'CBOE Volatility Index (VIX)',

    # Inflation
    'CPIAUCSL': 'Consumer Price Index (CPI)',

    # Economic Growth
    'GDP': 'Gross Domestic Product',
    'GDPC1': 'Real Gross Domestic Product',
    'INDPRO': 'Industrial Production Index',
    'UNRATE': 'Unemployment Rate',
    'PAYEMS': 'Total Nonfarm Payrolls',

    # Consumer / Business Activity
    'RSAFS': 'Retail Sales',
    'PCE': 'Personal Consumption Expenditures',
    'HOUST': 'Housing Starts',
    'BUSLOANS': 'Commercial and Industrial Loans',

    # International Trade
    'NETEXP': 'Net Exports of Goods and Services',
    'DTWEXBGS': 'Trade Weighted U.S. Dollar Index (Broad)',

    # Commodity Prices
    'DCOILWTICO': 'Crude Oil Prices: WTI',
}

def fetch_macro_data(api_key, start_date, end_date, output_dir='./macro_data', frequency='D'):
    """
    Fetch macroeconomic data from FRED

    Parameters:
    -----------
    api_key : str
        FRED API key
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    output_dir : str
        Directory to save output files
    frequency : str
        Data frequency ('D' for daily, 'M' for monthly, 'Q' for quarterly)

    Returns:
    --------
    DataFrame
        DataFrame with all macroeconomic indicators
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Check if the API key is provided
    if api_key and api_key.strip() != '':
        try:
            fred_api = fred.Fred(api_key=api_key)
            # Verify if the API key is valid by making a test request
            test_series = fred_api.get_series('DGS10', observation_start='2020-01-01', observation_end='2020-01-02')
            use_fred_api = True
        except Exception as e:
            print(f"Error using FRED API: {str(e)}")
            use_fred_api = False
    else:
        use_fred_api = False

    # Adjust start date to pull more historical data for calculations
    adjusted_start = (datetime.strptime(start_date, '%Y-%m-%d') - relativedelta(years=1)).strftime('%Y-%m-%d')

    # Create empty DataFrame to store all indicators
    all_data = pd.DataFrame()

    # Fetch data for each indicator
    print(f"Fetching {len(MACRO_INDICATORS)} macroeconomic indicators...")
    for code, name in tqdm(MACRO_INDICATORS.items()):
        try:
            # Get data from FRED
            if use_fred_api:
                series = fred_api.get_series(code, observation_start=adjusted_start, observation_end=end_date)
                indicator_df = pd.DataFrame(series, columns=[code])
            else:
                # Use pandas_datareader as fallback
                indicator_df = web.DataReader(code, 'fred', adjusted_start, end_date)

            if all_data.empty:
                all_data = indicator_df
            else:
                # Join with existing data
                all_data = all_data.join(indicator_df, how='outer')

        except Exception as e:
            print(f"Error fetching {code} ({name}): {str(e)}")

    # Calculate derived indicators on raw data before any filling
    derived_data = calculate_derived_indicators(all_data)
    
    # Now handle frequency resampling
    if frequency != 'D':
        if frequency == 'B':
            # Business day resampling - only fill actual business days
            derived_data = derived_data.asfreq('B', method='ffill')
        else:
            derived_data = derived_data.resample(frequency).last()

    # Forward fill missing values (for daily data with less frequent updates)
    derived_data = derived_data.ffill()

    # Drop rows with missing dates from adjusted start period
    derived_data = derived_data[derived_data.index >= start_date]

    # Save to CSV
    output_path = os.path.join(output_dir, f'macro_indicators_{frequency}.csv')
    derived_data.to_csv(output_path)
    print(f"Saved {derived_data.shape[1]} macroeconomic indicators to {output_path}")

    return derived_data

def calculate_derived_indicators(data):
    """
    Calculate additional derived indicators

    Parameters:
    -----------
    data : DataFrame
        DataFrame with raw macroeconomic indicators

    Returns:
    --------
    DataFrame
        DataFrame with additional derived indicators
    """
    df = data.copy()

    # Calculate yield curve indicators if not already present
    if 'DGS10' in df.columns and 'DGS2' in df.columns and 'T10Y2Y' not in df.columns:
        df['T10Y2Y'] = df['DGS10'] - df['DGS2']

    if 'DGS10' in df.columns and 'FEDFUNDS' in df.columns:
        df['T10YFFM'] = df['DGS10'] - df['FEDFUNDS']

    # Only calculate rate of change on actual data points (before any filling)
    # This preserves the true periodicity of data like CPI
    
    # Rate of change for inflation indicators
    if 'CPIAUCSL' in df.columns:
        # Monthly inflation rate - only computed where actual data exists
        valid_cpi = ~df['CPIAUCSL'].isna()
        df.loc[valid_cpi, 'CPI_MOM'] = df.loc[valid_cpi, 'CPIAUCSL'].pct_change() * 100
        # Annual inflation rate
        df.loc[valid_cpi, 'CPI_YOY'] = df.loc[valid_cpi, 'CPIAUCSL'].pct_change(periods=12) * 100

    # Calculate rate of change for growth indicators
    if 'GDPC1' in df.columns:
        valid_gdp = ~df['GDPC1'].isna()
        df.loc[valid_gdp, 'REAL_GDP_QOQ'] = df.loc[valid_gdp, 'GDPC1'].pct_change() * 100
        df.loc[valid_gdp, 'REAL_GDP_YOY'] = df.loc[valid_gdp, 'GDPC1'].pct_change(periods=4) * 100

    if 'INDPRO' in df.columns:
        valid_indpro = ~df['INDPRO'].isna()
        df.loc[valid_indpro, 'INDPRO_MOM'] = df.loc[valid_indpro, 'INDPRO'].pct_change() * 100
        df.loc[valid_indpro, 'INDPRO_YOY'] = df.loc[valid_indpro, 'INDPRO'].pct_change(periods=12) * 100

    if 'PAYEMS' in df.columns:
        valid_payems = ~df['PAYEMS'].isna()
        df.loc[valid_payems, 'PAYEMS_MOM'] = df.loc[valid_payems, 'PAYEMS'].pct_change() * 100
        df.loc[valid_payems, 'PAYEMS_YOY'] = df.loc[valid_payems, 'PAYEMS'].pct_change(periods=12) * 100

    # Calculate rate of change for commodity prices
    if 'DCOILWTICO' in df.columns:
        valid_oil = ~df['DCOILWTICO'].isna()
        df.loc[valid_oil, 'OIL_MOM'] = df.loc[valid_oil, 'DCOILWTICO'].pct_change() * 100
        df.loc[valid_oil, 'OIL_YOY'] = df.loc[valid_oil, 'DCOILWTICO'].pct_change(periods=252) * 100

    return df

def fetch_market_indices(start_date, end_date, output_dir='./macro_data'):
    """
    Fetch major market indices as additional macro indicators

    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    output_dir : str
        Directory to save output files

    Returns:
    --------
    DataFrame
        DataFrame with market indices
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Define major indices to track
    indices = {
        '^GSPC': 'S&P500',
        '^DJI': 'Dow_Jones',
        '^IXIC': 'Nasdaq',
        '^RUT': 'Russell_2000',
        '^VIX': 'VIX',
        '^TNX': 'Treasury_Yield_10yr',
        '^TYX': 'Treasury_Yield_30yr',
        '^FVX': 'Treasury_Yield_5yr',
        'GC=F': 'Gold_Futures',
        'CL=F': 'Crude_Oil_Futures',
        'ZB=F': 'Treasury_Bond_Futures',
        'ZN=F': 'Treasury_Note_Futures',
        'ZT=F': 'Treasury_2yr_Futures'
    }

    # Fetch data for each index
    all_indices = pd.DataFrame()

    print(f"Fetching {len(indices)} market indices from Yahoo Finance...")
    for ticker, name in tqdm(indices.items()):
        try:
            # Get data from Yahoo Finance
            index_data = yf.download(ticker, start=start_date, end=end_date, progress=False)

            # Extract the closing price
            index_df = pd.DataFrame(index_data['Close'])
            index_df.columns = [name]

            # If this is the first index, initialize the all_indices DataFrame
            if all_indices.empty:
                all_indices = index_df
            else:
                # Join with existing data
                all_indices = all_indices.join(index_df, how='outer')

        except Exception as e:
            print(f"Error fetching {ticker} ({name}): {str(e)}")

    # Calculate returns for each index
    for name in all_indices.columns:
        all_indices[f'{name}_daily_return'] = all_indices[name].pct_change(fill_method=None) * 100
        all_indices[f'{name}_daily_return'] = all_indices[f'{name}_daily_return'].ffill().bfill()

    # Fill any remaining missing values
    all_indices = all_indices.ffill().bfill()

    # Save to CSV
    output_path = os.path.join(output_dir, 'market_indices.csv')
    all_indices.to_csv(output_path)
    print(f"Saved {len(indices)} market indices to {output_path}")

    return all_indices

def combine_macro_data(fred_data, market_data, output_dir='./macro_data'):
    """
    Combine FRED data and market indices

    Parameters:
    -----------
    fred_data : DataFrame
        DataFrame with FRED indicators
    market_data : DataFrame
        DataFrame with market indices
    output_dir : str
        Directory to save output files

    Returns:
    --------
    DataFrame
        Combined DataFrame
    """
    # output directory
    os.makedirs(output_dir, exist_ok=True)

    # Ensure both datasets have a datetime index
    if not isinstance(fred_data.index, pd.DatetimeIndex):
        fred_data.index = pd.to_datetime(fred_data.index)
    
    if not isinstance(market_data.index, pd.DatetimeIndex):
        market_data.index = pd.to_datetime(market_data.index)

    # Merge the datasets
    combined_data = fred_data.join(market_data, how='outer')

    # Fill missing values using appropriate methods based on frequency. First we identify the common frequency:
    if combined_data.index.inferred_freq == 'D' or combined_data.index.inferred_freq == 'B':
        # Daily data forward fill for weekends/holidays, then backward fill any remaining gaps
        combined_data = combined_data.ffill(limit=5).bfill()
    elif combined_data.index.inferred_freq == 'M':
        # Monthly data forward fill for skipped months
        combined_data = combined_data.ffill().bfill()
    else:
        # Mixed or uncertain frequency
        combined_data = combined_data.ffill(limit=30).bfill(limit=5)

    # Handle remaining NaNs (if any)
    if combined_data.isna().any().any():
        print("Warning: Some NaN values remain in the combined dataset")
        # Fill with column means as a last resort
        for col in combined_data.columns:
            if combined_data[col].isna().any():
                col_mean = combined_data[col].mean()
                combined_data[col] = combined_data[col].fillna(col_mean)

    # Save to CSV
    output_path = os.path.join(output_dir, 'combined_macro_data.csv')
    combined_data.to_csv(output_path)
    print(f"Saved combined macro data with {combined_data.shape[1]} features to {output_path}")

    return combined_data

def align_macro_with_stock_data(stock_data, macro_data):
    """
    Align macro data with stock data with better handling of forward fills
    """
    # Ensure both datasets have datetime indices
    if not isinstance(stock_data.index, pd.DatetimeIndex):
        stock_data.index = pd.to_datetime(stock_data.index)
    
    if not isinstance(macro_data.index, pd.DatetimeIndex):
        macro_data.index = pd.to_datetime(macro_data.index)
    
    # If macro data is empty, return just the stock data
    if macro_data.empty:
        return stock_data.copy()
    
    # Create a copy of macro data to work with
    daily_macro = macro_data.copy()
    
    # Add "is_actual" flag columns for each macro indicator
    actual_cols = {}
    for col in daily_macro.columns:
        actual_col_name = f"is_actual_{col}"
        daily_macro[actual_col_name] = ~daily_macro[col].isna()
        actual_cols[col] = actual_col_name
    
    # Apply different forward fill limits based on data frequency
    for col in daily_macro.columns:
        if col.startswith('is_actual_'):
            continue
            
        # Monthly data (like CPI, GDP)
        if col in ['CPIAUCSL', 'GDP', 'GDPC1', 'PCE']:
            daily_macro[col] = daily_macro[col].ffill(limit=3)
        # Weekly data
        elif col in ['UNRATE', 'INDPRO', 'HOUST', 'NETEXP']:
            daily_macro[col] = daily_macro[col].ffill(limit=5)
        # Daily data (market indices, rates)
        else:
            daily_macro[col] = daily_macro[col].ffill(limit=2)
    
    # Resample to business days to match stock data
    daily_macro = daily_macro.asfreq('B', method=None)
    
    # Rename columns to add the MACRO_ prefix
    renamed_cols = {}
    for col in daily_macro.columns:
        if col.startswith('is_actual_'):
            renamed_cols[col] = f"MACRO_{col}"
        else:
            renamed_cols[col] = f"MACRO_{col}"
    daily_macro = daily_macro.rename(columns=renamed_cols)
    
    # Merge data
    aligned_data = pd.merge(
        stock_data, 
        daily_macro,
        left_index=True, 
        right_index=True,
        how='left'  # Keep all stock data rows
    )
    
    return aligned_data

def generate_aligned_charts_and_data(tickers, start_date, end_date, 
                                   window_size_days=14, step_size_days=7,
                                   output_dir='./aligned_data', 
                                   fred_api_key=None):
    """
    Comprehensive function to generate stock charts and data
    with properly aligned macroeconomic indicators

    Parameters:
    -----------
    tickers : list
        List of stock tickers
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    window_size_days : int
        Size of each time window in days
    step_size_days : int
        Step size between windows in days
    output_dir : str
        Output directory for all data
    fred_api_key : str
        FRED API key

    Returns:
    --------
    dict
        Dictionary with all generated data
    """
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories
    stock_dir = os.path.join(output_dir, 'stock_data')
    macro_dir = os.path.join(output_dir, 'macro_data')
    chart_dir = os.path.join(output_dir, 'charts')
    aligned_dir = os.path.join(output_dir, 'aligned_data')
    
    os.makedirs(stock_dir, exist_ok=True)
    os.makedirs(macro_dir, exist_ok=True)
    os.makedirs(chart_dir, exist_ok=True)
    os.makedirs(aligned_dir, exist_ok=True)
    
    # Fetch and process macroeconomic data
    print("\n--- Fetching Macroeconomic Data ---")
    # Get FRED macroeconomic indicators
    macro_data = fetch_macro_data(fred_api_key, start_date, end_date, macro_dir)
    
    # Get market indices
    market_data = fetch_market_indices(start_date, end_date, macro_dir)
    
    # Combine macro and market data
    combined_macro = combine_macro_data(macro_data, market_data, macro_dir)
    
    # Process stock data for each ticker
    print("\n--- Processing Stock Data ---")
    all_stock_data = {}
    all_windows = []
    all_charts = []
    
    for ticker in tqdm(tickers, desc="Processing tickers"):
        try:
            # Download and process stock data
            print(f"\nProcessing {ticker}...")
            
            # Get stock price data
            stock_data = get_stock_data(ticker, start_date, end_date)
            
            # Save the raw stock data
            stock_file = os.path.join(stock_dir, f"{ticker}_prices.csv")
            stock_data.to_csv(stock_file)
            
            # Calculate technical indicators
            tech_indicators = calculate_technical_indicators(stock_data)
            indicators_file = os.path.join(stock_dir, f"{ticker}_indicators.csv")
            tech_indicators.to_csv(indicators_file)
            
            # Create time windows
            windows = create_time_windows(tech_indicators, window_size_days, step_size_days)
            print(f"Created {len(windows)} time windows for {ticker}")
            
            # Store for later processing
            all_windows.extend([(ticker, window) for window in windows])
            
            # Store stock data
            all_stock_data[ticker] = {
                'prices': stock_data,
                'indicators': tech_indicators,
                'windows': windows
            }
            
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
    
    # Create aligned data for each window and generate charts
    print("\n--- Creating Aligned Data and Charts ---")
    
    # Track all generated data
    all_aligned_data = []
    
    for ticker, window in tqdm(all_windows, desc="Generating aligned data and charts"):
        try:
            window_start, window_end = window
            
            # Get stock data for this window
            stock_data = all_stock_data[ticker]['prices']
            window_stock = stock_data[(stock_data.index >= window_start) & 
                                     (stock_data.index <= window_end)]
            
            # Get indicators for this window
            tech_indicators = all_stock_data[ticker]['indicators']
            window_indicators = tech_indicators[(tech_indicators.index >= window_start) & 
                                              (tech_indicators.index <= window_end)]
            
            # Get macro data for this window
            window_macro = combined_macro[(combined_macro.index >= window_start) & 
                                       (combined_macro.index <= window_end)]
            
            # Prepare all DataFrame components separately
            
            window_stock_copy = window_stock.copy()
            
            if not window_macro.empty:
                # Add prefix to all columns
                renamed_macro_cols = {col: f'MACRO_{col}' for col in window_macro.columns}
                window_macro_renamed = window_macro.rename(columns=renamed_macro_cols)
            else:
                window_macro_renamed = pd.DataFrame(index=window_stock_copy.index)
            
            # Prepare technical indicators 
            tech_cols = [col for col in window_indicators.columns if col not in window_stock_copy.columns]
            window_tech = window_indicators[tech_cols] if tech_cols else pd.DataFrame(index=window_stock_copy.index)
            
            # Combine all three components at once using merge operations
            # First, merge stock and macro data
            aligned_data = pd.merge(
                window_stock_copy,
                window_macro_renamed,
                left_index=True,
                right_index=True,
                how='left'
            )
            
            # Then, merge with technical indicators
            if not window_tech.empty:
                aligned_data = pd.merge(
                    aligned_data,
                    window_tech,
                    left_index=True,
                    right_index=True,
                    how='left'
                )
            
            # Fill any NaN values in feature columns only, not target columns
            feature_cols = [col for col in aligned_data.columns if not col.startswith('Return_')]
            aligned_data[feature_cols] = aligned_data[feature_cols].ffill()
            
            # Calculate return for this window
            if len(window_stock) > 1:
                window_return = (window_stock['Close'].iloc[-1] / window_stock['Close'].iloc[0] - 1) * 100
            else:
                window_return = 0
            
            # Create window identifier
            window_id = f"{ticker}_{window_start.strftime('%Y%m%d')}_{window_end.strftime('%Y%m%d')}"
            
            # Create chart dir for this ticker
            ticker_chart_dir = os.path.join(chart_dir, ticker)
            os.makedirs(ticker_chart_dir, exist_ok=True)
            
            # Generate chart
            chart_file = os.path.join(ticker_chart_dir, f"{window_id}.png")
            chart_success = generate_candlestick_chart(stock_data, ticker, window, chart_file)
            
            # Save aligned data
            aligned_file = os.path.join(aligned_dir, f"{window_id}_aligned.csv")
            aligned_data.to_csv(aligned_file)
            
            # Record everything
            if chart_success:
                data_point = {
                    'ticker': ticker,
                    'window_id': window_id,
                    'start_date': window_start.strftime('%Y-%m-%d'),
                    'end_date': window_end.strftime('%Y-%m-%d'),
                    'chart_path': chart_file,
                    'data_path': aligned_file,
                    'window_return': window_return,
                    'has_chart': True
                }
                all_charts.append(data_point)
                all_aligned_data.append(data_point)
            
        except Exception as e:
            print(f"Error processing window for {ticker}: {str(e)}")
    
    # Create consolidated datasets for ML
    print("\n--- Creating Consolidated ML Datasets ---")
    
    # Convert to DataFrame and save
    charts_df = pd.DataFrame(all_charts)
    charts_path = os.path.join(output_dir, 'all_charts.csv')
    charts_df.to_csv(charts_path, index=False)
    
    aligned_df = pd.DataFrame(all_aligned_data)
    aligned_path = os.path.join(output_dir, 'all_aligned_data.csv')
    aligned_df.to_csv(aligned_path, index=False)
    
    # Create a consolidated ML-ready dataset
    consolidated_data = create_consolidated_ml_dataset(aligned_df, aligned_dir, output_dir)
    
    print(f"\nProcessing complete!")
    print(f"Total charts generated: {len(all_charts)}")
    print(f"Total aligned datasets: {len(all_aligned_data)}")
    print(f"Results saved to: {output_dir}")
    
    return {
        'stock_data': all_stock_data,
        'macro_data': combined_macro,
        'charts': charts_df,
        'aligned_data': aligned_df,
        'ml_dataset': consolidated_data
    }

def create_consolidated_ml_dataset(aligned_data_info, aligned_dir, output_dir):
    """
    Create a consolidated dataset ready for machine learning training
    
    Parameters:
    -----------
    aligned_data_info : DataFrame
        DataFrame with information about all aligned data files
    aligned_dir : str
        Directory containing aligned data files
    output_dir : str
        Directory to save output files
        
    Returns:
    --------
    DataFrame
        Consolidated dataset ready for ML
    """
    print("Creating consolidated ML dataset...")
    # List to store all data points
    all_data_points = []
    
    # Process each aligned data file
    for _, row in tqdm(aligned_data_info.iterrows(), total=len(aligned_data_info)):
        try:
            # Load the aligned data
            aligned_file = row['data_path']
            if os.path.exists(aligned_file):
                aligned_data = pd.read_csv(aligned_file, index_col=0, parse_dates=True)
                
                # Create a data point with key features
                data_point = {
                    'ticker': row['ticker'],
                    'window_id': row['window_id'],
                    'start_date': row['start_date'],
                    'end_date': row['end_date'],
                    'chart_path': row['chart_path'],
                    'window_return': float(row['window_return'])
                }
                
                # Get the last day's data for features
                if not aligned_data.empty:
                    last_day = aligned_data.iloc[-1]
                    
                    # Collect all features first, then add them at once
                    feature_dict = {}
                    
                    # Add technical indicators as features
                    tech_indicators = ['RSI', 'MACD', 'BB_Upper', 'BB_Lower', 
                                      'Volatility_20d', 'SMA_7', 'SMA_14', 'SMA_21']
                    for indicator in tech_indicators:
                        if indicator in last_day.index:
                            feature_dict[f'TECH_{indicator}'] = float(last_day[indicator])
                    
                    # Add macro and lagged returns indicators as features
                    for col in last_day.index:
                        if col.startswith(('MACRO_', 'Lagged_')):
                            try:
                                feature_dict[col] = float(last_day[col])
                            except (ValueError, TypeError):
                                # Skip any values that can't be converted to float
                                pass
                    
                    # Add target variables 
                    target_cols = ['Return_1d', 'Return_5d', 'Return_10d']
                    for col in target_cols:
                        if col in last_day.index:
                            try:
                                feature_dict[f'TARGET_{col}'] = float(last_day[col])
                            except (ValueError, TypeError):
                                # Skip any values that can't be converted to float
                                pass
                    
                    # Update data point with all features at once
                    data_point.update(feature_dict)
                
                # Add to list only if all target variables are present
                has_all_targets = all(f'TARGET_{col}' in data_point for col in target_cols)
                if has_all_targets:
                    all_data_points.append(data_point)
            
        except Exception as e:
            print(f"Error processing {row['window_id']}: {str(e)}")
    
    # Convert to DataFrame
    ml_dataset = pd.DataFrame(all_data_points)
    
    # Handle missing values for features only (not targets)
    feature_cols = [col for col in ml_dataset.select_dtypes(include=[np.number]).columns 
                   if not col.startswith('TARGET_')]
    
    # Calculate means for numeric feature columns only
    col_means = ml_dataset[feature_cols].mean()
    
    # Fill NaN values in numeric feature columns with their respective means
    for col in feature_cols:
        ml_dataset[col] = ml_dataset[col].fillna(col_means[col])
    
    # Non-numeric columns - fill with appropriate values
    for col in ml_dataset.columns:
        if col not in ml_dataset.select_dtypes(include=[np.number]).columns:
            if col == 'ticker' or col == 'window_id':
                # These are string columns - fill with 'unknown'
                ml_dataset[col] = ml_dataset[col].fillna('unknown')
            elif col == 'start_date' or col == 'end_date':
                # These are date columns - leave as is
                pass
            elif col == 'chart_path':
                # Fill with empty string
                ml_dataset[col] = ml_dataset[col].fillna('')
    
    # Save to CSV
    ml_path = os.path.join(output_dir, 'ml_dataset.csv')
    ml_dataset.to_csv(ml_path, index=False)
    print(f"Saved ML dataset with {len(ml_dataset)} samples to {ml_path}")
    
    # Split into train/val/test sets
    train_data, val_data, test_data = split_dataset(ml_dataset)
    
    # Save splits
    train_path = os.path.join(output_dir, 'train_dataset.csv')
    val_path = os.path.join(output_dir, 'val_dataset.csv')
    test_path = os.path.join(output_dir, 'test_dataset.csv')
    
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)
    test_data.to_csv(test_path, index=False)
    
    print(f"Split dataset: {len(train_data)} training, {len(val_data)} validation, {len(test_data)} test samples")
    
    return ml_dataset

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, temporal=True):
    """
    Split dataset into train, validation, and test sets
    with option for temporal (time-based) splitting
    
    Parameters:
    -----------
    dataset : DataFrame
        Dataset to split
    train_ratio : float
        Ratio for training set
    val_ratio : float
        Ratio for validation set
    test_ratio : float
        Ratio for test set
    temporal : bool
        Whether to split temporally (by date) or randomly
        
    Returns:
    --------
    tuple
        (train_data, val_data, test_data)
    """
    if temporal:
        # Sort by end_date to maintain temporal order
        if 'end_date' in dataset.columns:
            dataset = dataset.sort_values('end_date')
        
        # Calculate split indices
        n = len(dataset)
        train_idx = int(n * train_ratio)
        val_idx = train_idx + int(n * val_ratio)
        
        # Split the data
        train_data = dataset.iloc[:train_idx].copy()
        val_data = dataset.iloc[train_idx:val_idx].copy()
        test_data = dataset.iloc[val_idx:].copy()
    else:
        # Random splitting
        from sklearn.model_selection import train_test_split
        
        # First split into train and temp
        train_data, temp_data = train_test_split(
            dataset, 
            test_size=(val_ratio + test_ratio),
            random_state=42
        )
        
        # Then split temp into val and test
        val_data, test_data = train_test_split(
            temp_data,
            test_size=test_ratio/(val_ratio + test_ratio),
            random_state=42
        )
    
    return train_data, val_data, test_data

def run_integrated_pipeline(tickers=None, start_date='2005-01-01', end_date='2023-12-31',
                          window_size_days=14, step_size_days=7, 
                          output_dir='./integrated_data',
                          fred_api_key=None, use_custom_indicators=True):
    """
    Run the complete integrated pipeline for stock and macro data
    
    Parameters:
    -----------
    tickers : list
        List of stock tickers (default: None, will use a predefined list)
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    window_size_days : int
        Size of each time window in days
    step_size_days : int
        Step size between windows in days
    output_dir : str
        Output directory for all data
    fred_api_key : str
        FRED API key (optional)
        
    Returns:
    --------
    dict
        Dictionary with all generated data
    """
    # Define default tickers if none provided
    if tickers is None:
        tickers=[
        # Technology
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
        # Finance
        'JPM', 'BAC', 'GS', 'C', 'MS',
        # Healthcare
        'JNJ', 'PFE', 'UNH', 'MRK', 'ABBV',
        # Consumer
        'PG', 'KO', 'PEP', 'WMT', 'MCD',
        # Energy
        'XOM', 'CVX', 'COP', 'BP', 'SLB']
        
    # Run the integrated pipeline
    results = generate_aligned_charts_and_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        window_size_days=window_size_days,
        step_size_days=step_size_days,
        output_dir=output_dir,
        fred_api_key=fred_api_key
    )
    
    return results

if __name__ == "__main__":
    fred_api_key = "YOUR_FRED_API_KEY_HERE"  
        
    # Define parameters
    start_date = '2005-01-01'  
    end_date = '2023-12-31'   
    window_size_days = 14      # 14-day windows
    step_size_days = 7         # Move forward 7 days each time
    output_dir = './integrated_stock_macro_data'  # Output directory
    
    # Run the pipeline directly (not through the main function)
    print(f"Starting pipeline with API key: {fred_api_key}")
    
    # Run directly with generate_aligned_charts_and_data to minimize potential issues
    results = generate_aligned_charts_and_data(
        tickers=[
        # Technology
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
        # Finance
        'JPM', 'BAC', 'GS', 'C', 'MS',
        # Healthcare
        'JNJ', 'PFE', 'UNH', 'MRK', 'ABBV',
        # Consumer
        'PG', 'KO', 'PEP', 'WMT', 'MCD',
        # Energy
        'XOM', 'CVX', 'COP', 'BP', 'SLB'], 
        start_date=start_date,
        end_date=end_date,
        window_size_days=window_size_days,
        step_size_days=step_size_days,
        output_dir=output_dir,
        fred_api_key=fred_api_key
    )


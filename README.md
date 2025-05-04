# Stock Market Return Prediction with Multimodal Deep Learning

This repository contains a multimodal deep learning approach to predict stock market returns using both visual (candlestick charts) and tabular (technical indicators, macroeconomic data) features.

## Project Overview

The model combines:
- **Vision Transformer (ViT)** for processing candlestick chart images
- **Tabular Transformer** for processing time-series technical indicators and macroeconomic data
- **Cross-modal attention mechanism** to fuse information from both modalities

The model predicts directional movement (up/down) for three time horizons:
- 1-day ahead (next day)
- 5-days ahead (one week)
- 10-days ahead (two weeks)

## Requirements

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.6.13
pandas>=1.5.3
numpy>=1.24.2
matplotlib>=3.7.1
scikit-learn>=1.2.2
tqdm>=4.65.0
Pillow>=9.5.0
mplfinance>=0.12.9b7
yfinance>=0.2.18
fredapi>=0.5.1
pandas-datareader>=0.10.0
ipywidgets>=8.0.6
seaborn>=0.12.2
python-dateutil>=2.8.2
typing-extensions>=4.5.0
```

You can install all dependencies using:

```bash
pip install -r requirements.txt
```

Create a requirements.txt file with the above packages or use the one provided in the repository.

## Project Structure

```
.
├── configs_init.py      # Configuration module initialization
├── configs.py           # Model and training configurations
├── data_generation.py   # Data acquisition and preprocessing
├── data_init.py         # Dataset module initialization
├── data.py              # Dataset class implementation
├── main.py              # Training script
├── models_encoders.py   # Feature encoders (ViT, TabTransformer)
├── models_fusion.py     # Cross-modal fusion
├── models_init.py       # Model module initialization
├── models.py            # Main model architecture
├── testing.py           # Model evaluation script
├── training_init.py     # Training module initialization
├── training_loss.py     # Loss functions
├── training_utils.py    # Training utilities
└── training.py          # Training and evaluation loops
```

## Setup and Data Generation

### Step 1: Data Generation

The first step is to generate the dataset which includes downloading stock price data, creating candlestick charts, and collecting macroeconomic indicators.

```bash
# Install required dependencies
pip install -r requirements.txt

# Run data generation script
python data_generation.py
```

This script performs the following steps:
1. Downloads stock price data for specified tickers from Yahoo Finance
2. Downloads macroeconomic indicators from FRED (Federal Reserve Economic Data)
3. Calculates technical indicators for each stock (RSI, MACD, Bollinger Bands, etc.)
4. Generates candlestick chart images for specified time windows
5. Aligns stock data with macroeconomic data
6. Creates and saves train/validation/test datasets

By default, the script processes data for the following stocks:
- Technology: AAPL, MSFT, GOOGL, AMZN, META
- Finance: JPM, BAC, GS, C, MS
- Healthcare: JNJ, PFE, UNH, MRK, ABBV
- Consumer: PG, KO, PEP, WMT, MCD
- Energy: XOM, CVX, COP, BP, SLB

You can modify the list of tickers in the `run_integrated_pipeline` function at the bottom of the script.

The data generation process takes several hours, depending on the number of tickers and date range.

**Important Note:** The script requires a FRED API key to download macroeconomic data. In the code, you'll need to replace the placeholder API key with your own:

```python
# In data_generation.py, find this line:
fred_api_key = "YOUR_FRED_API_KEY_HERE"  # Replace with your actual API key
```

You can obtain a free API key by registering at https://fred.stlouisfed.org/docs/api/api_key.html

By default, the data is saved to the `./integrated_stock_macro_data` directory, which will contain:
- Candlestick chart images
- Train/validation/test CSV files
- Raw and processed data files

### Step 2: Model Training

Once the data is generated, you can train the model using:

```bash
# Run training script
python main.py
```

The training configuration can be modified in `configs.py`. Key parameters include:
- Batch size
- Learning rate
- Number of epochs
- Early stopping patience
- Model architecture parameters

During training, the model checkpoints and metrics will be saved to the `./output` directory (configurable in `configs.py`).

### Step 3: Model Evaluation

After training, evaluate the model on the test set:

```bash
# Run evaluation script
python testing.py
```

This will load the best model checkpoint and compute evaluation metrics, including:
- Directional accuracy (up/down prediction)
- Precision, recall, and F1 score
- Confusion matrices

Results will be saved to `./output/test_results/`.

## Model Architecture

The model architecture consists of three main components:

1. **Image Encoder (ViT)**: Processes candlestick chart images to extract visual patterns.
2. **Tabular Encoder (Transformer)**: Processes time-series data including technical indicators and macroeconomic features.
3. **Cross-Modal Fusion**: Combines information from both modalities using attention mechanisms.

The prediction head outputs directional probabilities for three time horizons (1-day, 5-day, 10-day).

## Dataset

The dataset contains data for 25 major US stocks across various sectors (Technology, Finance, Healthcare, Consumer, Energy) from 2005 to 2023.

Each data point includes:
- Candlestick chart image for a 14-day window
- Time-series technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Macroeconomic indicators from FRED (interest rates, GDP, inflation, etc.)
- Market indices (S&P 500, NASDAQ, etc.)
- Target variables: Return for 1-day, 5-day, and 10-day horizons

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Vision Transformer (ViT) implementation is based on the timm library.
- Macroeconomic data is obtained from FRED (Federal Reserve Economic Data).
- Stock price data is obtained from Yahoo Finance.


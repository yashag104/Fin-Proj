# Bitcoin Price Prediction - CNN-LSTM Hybrid Model

## 🎯 Project Overview

This project implements a **CNN-LSTM Hybrid Deep Learning Model** for Bitcoin price prediction using multi-timeframe analysis (1-minute and 5-minute data) with DSP preprocessing.

### Key Features:
- ✅ **DSP Denoising** - Savitzky-Golay filter to remove noise while preserving trends
- ✅ **Multi-timeframe Architecture** - Dual-path CNN for 1-min and 5-min data
- ✅ **CNN-LSTM Hybrid** - CNN for feature extraction, LSTM for sequence learning
- ✅ **Financial ML Best Practices** - Log returns, no data leakage, temporal train-test split
- ✅ **Comprehensive Evaluation** - RMSE, Directional Accuracy, Sharpe Ratio, Maximum Drawdown

---

## 🏗️ Model Architecture

```
[1-Minute Data] → CNN (3 Conv blocks) → ┐
                                         ├→ Concatenate → LSTM (2 layers) → Dense → Price Prediction
[5-Minute Data] → CNN (3 Conv blocks) → ┘
```

### Flow:
1. **Stage 1: Input** - Raw Bitcoin OHLCV data (1699 entries)
2. **Stage 2: DSP Preprocessing** - Savitzky-Golay filtering for denoising
3. **Stage 3: Feature Extraction (CNN)** - Detect local patterns with 1D convolutions
4. **Stage 4: Memory & Trend (LSTM)** - Learn temporal dependencies
5. **Stage 5: Output** - Final price prediction with Huber loss

---

## 📋 Requirements

### Python Version:
- Python 3.8 or higher

### Required Libraries:
```bash
tensorflow>=2.10.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
scipy>=1.7.0
matplotlib>=3.4.0
seaborn>=0.11.0
ta>=0.10.0
```

---

## 🚀 Quick Start Guide

### Step 1: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install tensorflow numpy pandas scikit-learn scipy matplotlib seaborn ta
```

### Step 2: Prepare Your Data

Place your Bitcoin CSV file in the project directory. The CSV should have columns like:
- `timestamp` / `date` / `timeOpen`
- `close` / `Close` / `Price`
- `open`, `high`, `low`, `volume` (optional)

**Example CSV formats supported:**
- Semicolon-separated (`;`)
- Comma-separated (`,`)
- Tab-separated

### Step 3: Run Data Preprocessing

```bash
python data_preprocessing.py
```

**What this does:**
- Loads and cleans Bitcoin data
- Applies Savitzky-Golay DSP filter
- Calculates log returns and technical indicators
- Creates 1-min and 5-min timeframes
- Generates sliding windows (60 timesteps)
- Performs temporal train-test split (80/20)

**Output files:**
- `X_1min_train.npy`, `X_1min_test.npy`
- `X_5min_train.npy`, `X_5min_test.npy`
- `y_price_train.npy`, `y_price_test.npy`
- `y_direction_train.npy`, `y_direction_test.npy`

### Step 4: Train the Model

```bash
python train.py
```

**Training configuration:**
- Epochs: 100 (with early stopping)
- Batch size: 16
- Loss function: Huber loss (robust to outliers)
- Optimizer: Adam (lr=0.001)
- Validation split: 10%

**What this does:**
- Builds CNN-LSTM hybrid model
- Trains with early stopping and learning rate reduction
- Evaluates on test set
- Calculates comprehensive metrics
- Generates visualization plots

**Output files:**
- `best_model.keras` - Best model during training
- `final_model.keras` - Final trained model
- `training_history.png` - Training curves
- `predictions.png` - Actual vs Predicted prices
- `directional_analysis.png` - Direction prediction accuracy
- `evaluation_results.npy` - Performance metrics

---

## 📊 Evaluation Metrics

### 1. Price Accuracy:
- **RMSE** (Root Mean Squared Error) - Lower is better
- **MAE** (Mean Absolute Error) - Lower is better
- **MAPE** (Mean Absolute Percentage Error) - Lower is better

### 2. Directional Accuracy:
- **DA** (Directional Accuracy) - % of correct up/down predictions
- Target: > 55% (above random baseline of 50%)

### 3. Trading Performance:
- **Sharpe Ratio** - Risk-adjusted returns (higher is better)
- **Maximum Drawdown** - Worst peak-to-trough decline (lower is better)

### 4. Walk-Forward Validation:
- Mimics real-life trading by predicting one step at a time
- More realistic performance estimate

---

## 📁 Project Structure

```
bitcoin_predictor/
│
├── data_preprocessing.py    # Data loading, DSP filtering, windowing
├── model.py                 # CNN-LSTM architecture definition
├── train.py                 # Training and evaluation script
├── README.md               # This file
│
├── your_bitcoin_data.csv   # Your Bitcoin CSV file (you provide)
│
├── X_1min_train.npy        # Generated after preprocessing
├── X_5min_train.npy
├── y_price_train.npy
├── ...
│
├── best_model.keras        # Generated after training
├── final_model.keras
├── training_history.png
├── predictions.png
└── directional_analysis.png
```

---

## 🔬 How It Works

### Why CNN-LSTM Hybrid?

**1. CNN (Convolutional Neural Network):**
- Extracts **local patterns** from time-series data
- Detects features like: spikes, trends, consolidations
- Acts as automatic feature extractor

**2. LSTM (Long Short-Term Memory):**
- Learns **temporal dependencies** and sequences
- Remembers that "Pattern A → Pattern B → Price C"
- Captures long-term trends and market memory

**3. Multi-Timeframe Strategy:**
- **1-minute window (The Microscope):** Captures volatility and immediate reactions
- **5-minute window (The Map):** Captures trends and filters out noise
- Combined: Best of both worlds!

### Why DSP Preprocessing?

- Financial data is **extremely noisy**
- Savitzky-Golay filter smooths noise while **preserving peaks**
- Helps the model focus on real trends, not random fluctuations

### Why Log Returns?

- Makes data **stationary** (constant mean over time)
- Easier for neural networks to learn patterns
- More appropriate for financial modeling

### Why No Shuffle?

- In time-series, **order matters**!
- Shuffling would cause **data leakage** (model sees the future)
- We use **temporal split**: train on past, test on future

---

## 🎛️ Configuration Options

### Modify Window Size:

In `data_preprocessing.py`:
```python
processor = EnhancedBitcoinProcessor(
    csv_path=csv_file,
    window_size=60,  # Change this (default: 60)
    test_split=0.2   # 20% test data
)
```

### Change Loss Function:

In `train.py`:
```python
model_builder.compile_model(
    learning_rate=0.001,
    loss_type='huber'  # Options: 'huber', 'mse', 'directional'
)
```

### Adjust Training Parameters:

In `train.py`:
```python
EPOCHS = 100
BATCH_SIZE = 16  # Try: 8, 16, 32
VALIDATION_SPLIT = 0.1
```

---

## 📈 Expected Results

For a well-trained model on 1699 Bitcoin data points:

### Good Performance:
- ✅ Directional Accuracy: **55-65%**
- ✅ RMSE: **< 5% of average price**
- ✅ Sharpe Ratio: **> 0**
- ✅ Training completes in **10-30 minutes** (depends on hardware)

### Warning Signs:
- ⚠️ Directional Accuracy < 52%: Model not learning patterns
- ⚠️ Very low loss but poor test accuracy: Overfitting
- ⚠️ Training loss not decreasing: Learning rate too high/low

---

## 🐛 Troubleshooting

### Issue 1: CSV Not Found
```bash
❌ No CSV file found!
```
**Solution:** Place your Bitcoin CSV file in the same directory as the scripts.

### Issue 2: Out of Memory
```bash
ResourceExhaustedError: OOM when allocating tensor
```
**Solution:** Reduce batch size in `train.py`:
```python
BATCH_SIZE = 8  # Instead of 16
```

### Issue 3: Poor Directional Accuracy
**Solutions:**
1. Increase window size: `window_size=90`
2. Try different loss: `loss_type='directional'`
3. Add more training data
4. Adjust DSP filter: `window_length=15`

### Issue 4: Model Not Improving
**Solutions:**
1. Check if data has trends (plot it first)
2. Increase epochs: `EPOCHS = 150`
3. Try lower learning rate: `learning_rate=0.0001`
4. Reduce model complexity (remove one LSTM layer)

---

## 📚 Understanding the Output

### 1. Training History Plot:
- **Loss curve:** Should decrease and stabilize
- **MAE/MSE:** Should converge
- **Directional Accuracy:** Should trend upward

### 2. Predictions Plot:
- **Blue line:** Actual prices
- **Red dashed:** Predicted prices
- **Should follow trend:** Model captures general direction

### 3. Directional Analysis:
- **Green dots:** Correct direction predictions
- **Red crosses:** Wrong predictions
- **Rolling accuracy:** Shows consistency over time

---

## 🎓 Key Concepts

### What is Directional Accuracy?
```python
If actual_price[t+1] > actual_price[t]:
    actual_direction = UP
If predicted_price[t+1] > predicted_price[t]:
    predicted_direction = UP

Directional Accuracy = % of times (actual_direction == predicted_direction)
```

### What is Sharpe Ratio?
```python
Sharpe Ratio = (Mean Return - Risk-Free Rate) / Std Dev of Returns

Higher Sharpe = Better risk-adjusted returns
> 1.0 = Good
> 2.0 = Excellent
```

### What is Maximum Drawdown?
```python
Max Drawdown = Largest peak-to-trough decline

Example: Price went $60,000 → $40,000 = 33% drawdown
Lower is better (less risk)
```

---

## 🔄 Next Steps

### To Improve Model:
1. **Add more data:** More historical Bitcoin data
2. **Feature engineering:** Add more technical indicators
3. **Hyperparameter tuning:** Try different architectures
4. **Ensemble methods:** Combine multiple models
5. **Attention mechanisms:** Add attention layers to LSTM

### To Deploy:
1. **Real-time prediction:** Connect to live Bitcoin API
2. **Trading bot:** Implement automated trading logic
3. **Risk management:** Add stop-loss and position sizing
4. **Backtesting:** Test on historical data with trading costs

---

## ⚠️ Disclaimer

This model is for **educational and research purposes only**. 

**Do NOT use for real trading without:**
- Extensive backtesting
- Risk management
- Understanding of financial markets
- Proper capital allocation

**Cryptocurrency trading is highly risky. Past performance does not guarantee future results.**

---

## 📞 Support

If you encounter issues:
1. Check the Troubleshooting section
2. Verify all dependencies are installed
3. Ensure CSV file format is correct
4. Check console output for specific error messages

---

## 📄 License

This project is provided as-is for educational purposes.

---

## 🙏 Acknowledgments

- TensorFlow/Keras for deep learning framework
- Technical Analysis Library (ta) for indicators
- SciPy for DSP filtering

---

**Happy Predicting! 🚀📈**

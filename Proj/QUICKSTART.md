# ⚡ QUICK START GUIDE

## 🚀 Get Running in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Add Your Bitcoin Data
Place your Bitcoin CSV file in this directory. Any of these formats work:
- `bitcoin_data.csv`
- `btc_prices.csv`
- Any CSV with timestamp and price columns

### Step 3: Run Everything
```bash
python run_all.py
```

That's it! The script will:
1. ✅ Preprocess your data with DSP filtering
2. ✅ Build and train the CNN-LSTM model
3. ✅ Generate evaluation metrics and plots
4. ✅ Save the trained model

⏱️ **Total time: ~30 minutes** (depends on your hardware)

---

## 📂 What Gets Created

After running, you'll have:

### Models:
- `best_model.keras` - Best model during training (use this!)
- `final_model.keras` - Final trained model

### Data Files:
- `X_1min_train.npy` - Processed 1-minute training data
- `X_5min_train.npy` - Processed 5-minute training data
- `X_1min_test.npy` - Test data (1-minute)
- `X_5min_test.npy` - Test data (5-minute)
- `y_price_train.npy` - Training labels (prices)
- `y_price_test.npy` - Test labels (prices)

### Visualizations:
- `training_history.png` - Training curves (loss, MAE, MSE)
- `predictions.png` - Actual vs Predicted prices
- `directional_analysis.png` - Direction prediction accuracy
- `model_architecture.png` - Model architecture diagram
- `process_flowchart.png` - Complete pipeline flowchart

### Metrics:
- `evaluation_results.npy` - All performance metrics saved

---

## 🎯 Alternative: Run Step-by-Step

If you prefer more control:

### 1. Preprocess Data
```bash
python data_preprocessing.py
```
This creates the `.npy` files from your CSV.

### 2. Train Model
```bash
python train.py
```
This trains the model and generates all visualizations.

### 3. Make Predictions (Optional)
```bash
python predict.py
```
This loads the trained model and makes predictions on test data.

### 4. Visualize Architecture (Optional)
```bash
python visualize_architecture.py
```
This generates architecture diagrams.

---

## 📊 Understanding Your Results

### Good Model Performance:
- ✅ **Directional Accuracy**: 55-65% (above 50% random baseline)
- ✅ **RMSE**: < 5% of average Bitcoin price
- ✅ **Sharpe Ratio**: > 0 (positive risk-adjusted returns)
- ✅ **Training Loss**: Decreases and stabilizes

### Warning Signs:
- ⚠️ **Directional Accuracy < 52%**: Model not learning patterns
- ⚠️ **Very high RMSE**: Model predictions are way off
- ⚠️ **Negative Sharpe**: Model would lose money in trading
- ⚠️ **Training loss not decreasing**: Learning rate issue

---

## 🔧 Troubleshooting

### Problem: "No CSV file found"
**Solution:** Place your Bitcoin CSV file in the same directory as these scripts.

### Problem: "Out of memory"
**Solution:** Open `train.py` and change:
```python
BATCH_SIZE = 8  # Instead of 16
```

### Problem: "Model not improving"
**Solution:** Try these in `data_preprocessing.py`:
```python
window_size=90  # Increase from 60
test_split=0.1  # Use more training data
```

### Problem: "Low directional accuracy"
**Solutions:**
1. Try directional loss in `train.py`:
   ```python
   model_builder.compile_model(loss_type='directional')
   ```
2. Increase training epochs
3. Get more historical data

---

## 📖 What Each File Does

### Core Scripts:
1. **data_preprocessing.py**
   - Loads Bitcoin CSV
   - Applies DSP filtering (Savitzky-Golay)
   - Calculates technical indicators
   - Creates sliding windows
   - Splits into train/test (no shuffle!)

2. **model.py**
   - Defines CNN-LSTM architecture
   - Implements Huber loss
   - Creates dual-path model for 1-min and 5-min data

3. **train.py**
   - Trains the model
   - Implements early stopping
   - Calculates all metrics
   - Generates visualizations

4. **predict.py**
   - Loads trained model
   - Makes predictions on test data
   - Visualizes results

5. **run_all.py**
   - Runs the complete pipeline
   - Automated execution

6. **visualize_architecture.py**
   - Generates architecture diagrams
   - Creates process flowcharts

---

## 🎓 Key Concepts

### Why CNN-LSTM?
- **CNN**: Extracts patterns (like spikes, trends) from time-series
- **LSTM**: Learns sequences (pattern A → pattern B → price C)
- **Together**: Best of both worlds!

### Why Two Timeframes (1-min & 5-min)?
- **1-min**: Captures volatility and immediate reactions (the microscope)
- **5-min**: Captures trends and filters noise (the map)
- **Combined**: Multi-scale analysis for better predictions

### Why DSP Filtering?
- Bitcoin data is **extremely noisy**
- Savitzky-Golay filter smooths noise but keeps trends
- Helps model focus on real patterns

### Why No Shuffle?
- In time-series, **order matters**!
- Shuffling = data leakage (seeing the future)
- We split temporally: train on past, test on future

---

## 📈 What to Expect

### With 1699 Bitcoin Data Points:

**Training:**
- Duration: 10-30 minutes
- Epochs: ~30-50 (with early stopping)
- Final train loss: 0.01-0.05
- Final val loss: 0.02-0.08

**Test Performance:**
- Directional Accuracy: 55-65%
- RMSE: $500-2000 (depends on price range)
- MAPE: 2-8%

**Files Generated:**
- 10+ output files
- Total size: ~50-200 MB

---

## 🔐 Important Notes

### ⚠️ DISCLAIMER
This model is for **educational purposes only**.

**DO NOT use for real trading without:**
- ✅ Extensive backtesting
- ✅ Risk management strategy
- ✅ Understanding of financial markets
- ✅ Proper capital allocation
- ✅ Stop-loss mechanisms

**Cryptocurrency trading is extremely risky!**

### 📊 Model Limitations
- Predicts next price, not long-term trends
- Accuracy degrades for multi-step predictions
- Sensitive to market regime changes
- Cannot predict black swan events
- Past performance ≠ future results

---

## 💡 Next Steps After Training

### 1. Analyze Results
- Check `training_history.png` for learning curves
- Review `predictions.png` for accuracy
- Study `directional_analysis.png` for trading potential

### 2. Improve Model
- Add more historical data
- Tune hyperparameters (window size, batch size)
- Try different architectures
- Add more technical indicators

### 3. Experiment
- Test on different cryptocurrencies
- Try different timeframes
- Implement ensemble methods
- Add attention mechanisms

### 4. For Real Use (Advanced)
- Implement walk-forward validation
- Add transaction costs
- Create risk management system
- Backtest thoroughly
- Paper trade first!

---

## 📞 Need Help?

1. **Check the README.md** - Comprehensive documentation
2. **Review error messages** - Usually self-explanatory
3. **Check file existence** - Make sure CSV file is present
4. **Verify dependencies** - Run: `pip install -r requirements.txt`

---

## ✅ Success Checklist

Before training, make sure:
- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Bitcoin CSV file in directory
- [ ] At least 1000+ data points in CSV
- [ ] Sufficient disk space (~500 MB free)
- [ ] Sufficient RAM (~4 GB free)

After training, you should have:
- [ ] `best_model.keras` file exists
- [ ] Training completed without errors
- [ ] Directional accuracy > 52%
- [ ] Visualization plots generated
- [ ] No memory errors

---

## 🎯 Expected Output Example

```
============================================================
BITCOIN PRICE PREDICTION - CNN-LSTM HYBRID TRAINING
============================================================

📂 Loading processed data...
✓ Data loaded successfully!
  Train samples: 1200
  Test samples: 300

🏗️  Building model...
✓ Model built with 500K parameters

🚀 Starting training...
Epoch 1/100: loss: 0.0234 - val_loss: 0.0312 - val_accuracy: 52.3%
Epoch 2/100: loss: 0.0198 - val_loss: 0.0289 - val_accuracy: 54.1%
...
Early stopping at epoch 42

✅ Training complete!

📊 Performance Metrics:
  RMSE: 1234.56
  MAE: 987.65
  MAPE: 3.45%
  Directional Accuracy: 58.23%

📈 Trading Metrics:
  Sharpe Ratio: 0.82
  Maximum Drawdown: -12.34%

✅ All files generated successfully!
```

---

**Ready to predict Bitcoin prices? Let's go! 🚀**

```bash
python run_all.py
```

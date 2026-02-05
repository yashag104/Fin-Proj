import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from scipy.signal import savgol_filter
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import warnings
import pickle
warnings.filterwarnings('ignore')


class EnhancedBitcoinProcessor:
    """
    Enhanced Bitcoin data processor with PROPER scaling (no data leakage)
    CUSTOM: Uses btc_15m_data_2018_to_2025.csv with 15-minute interval data
    """

    def __init__(self, csv_path='btc_15m_data_2018_to_2025.csv', window_size=60,
                 train_samples=None, test_samples=None, split_ratio=0.8):
        self.csv_path = csv_path
        self.window_size = window_size
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.split_ratio = split_ratio
        self.total_samples = 0  # Will be set later

        # CRITICAL FIX: Separate scalers for features and target
        self.feature_scaler_1min = MinMaxScaler(feature_range=(0, 1))
        self.feature_scaler_5min = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))  # NEW: For price only

    def load_and_clean_data(self):
        """Load and clean Bitcoin CSV data"""
        print("=" * 80)
        print("STEP 1: LOADING DATA")
        print("=" * 80)
        print(f"📂 Loading file: {self.csv_path}")

        df = None  # Defensive initialization

        # Try different separators
        for sep in [';', ',', '\t']:
            try:
                temp_df = pd.read_csv(self.csv_path, sep=sep)
                if len(temp_df.columns) > 1:
                    df = temp_df
                    break
            except Exception:
                continue

        if df is None:
            raise ValueError("❌ Failed to read CSV file with any supported separator (; , \\t)")

        print(f"✓ Raw data loaded: {df.shape}")
        print(f"✓ Columns: {df.columns.tolist()}")

        # --- NEW: Column names matching user's CSV ---
        # The CSV contains: Open time,Open,High,Low,Close,Volume,Close time,Quote asset volume,...
        timestamp_cols = [
            'Open time', 'Open Time', 'open time', 'open_time',
            'Close time', 'Close Time', 'close time', 'close_time',
            'timestamp', 'Timestamp', 'timeOpen', 'Date', 'date'
        ]
        close_cols = ['Close', 'close', 'Close_Price', 'Price']
        volume_cols = ['Volume', 'volume', 'Quote asset volume', 'quote asset volume', 'Quote Asset Volume']

        # Find matching actual column names (exact match required)
        timestamp_col = next((col for col in timestamp_cols if col in df.columns), None)
        close_col = next((col for col in close_cols if col in df.columns), None)
        volume_col = next((col for col in volume_cols if col in df.columns), None)

        if timestamp_col is None:
            raise ValueError("❌ Could not find timestamp column! (expected Open time or Close time or timestamp)")

        if close_col is None:
            raise ValueError("❌ Could not find price (Close) column! (expected 'Close')")

        # Parse timestamp (try ms, then s, then default)
        try:
            df['Datetime'] = pd.to_datetime(df[timestamp_col], unit='ms')
        except Exception:
            try:
                df['Datetime'] = pd.to_datetime(df[timestamp_col], unit='s')
            except Exception:
                df['Datetime'] = pd.to_datetime(df[timestamp_col])

        # Build cleaned DataFrame with required columns
        df_clean = pd.DataFrame()
        df_clean['Datetime'] = df['Datetime']
        df_clean['Close'] = pd.to_numeric(df[close_col], errors='coerce')

        # For Open/High/Low - these columns are present as 'Open','High','Low' in your CSV
        for col in ['Open', 'High', 'Low']:
            if col.lower() in [c.lower() for c in df.columns]:
                actual_col = next(c for c in df.columns if c.lower() == col.lower())
                df_clean[col] = pd.to_numeric(df[actual_col], errors='coerce')
            else:
                df_clean[col] = df_clean['Close']

        # Volume handling: use 'Volume' or 'Quote asset volume' if available
        if volume_col:
            df_clean['Volume'] = pd.to_numeric(df[volume_col], errors='coerce')
        else:
            # Default fallback
            df_clean['Volume'] = 1000000

        df_clean.set_index('Datetime', inplace=True)
        df_clean = df_clean.dropna(subset=['Close'])
        df_clean = df_clean.sort_index()

        print(f"✓ Cleaned data: {df_clean.shape}")
        print(f"✓ Date range: {df_clean.index.min()} to {df_clean.index.max()}")

        # Check if we have expected number of rows
        if len(df_clean) < self.total_samples:
            print(f"⚠️  Warning: Expected {self.total_samples} entries, got {len(df_clean)}")
            print(f"⚠️  Adjusting train/test split proportionally...")
            ratio = len(df_clean) / self.total_samples
            self.train_samples = int(self.train_samples * ratio)
            self.test_samples = len(df_clean) - self.train_samples

        return df_clean

    def apply_dsp_filtering(self, data, column='Close', window_length=11, polyorder=3):
        """Apply Savitzky-Golay filter - REDUCED filtering to preserve signal"""
        print("\n" + "=" * 80)
        print("STEP 2: DSP DENOISING (Savitzky-Golay Filter)")
        print("=" * 80)

        data = data.copy()

        # CRITICAL FIX: Use lighter filtering to preserve price movements
        window_length = min(window_length, len(data))
        if window_length % 2 == 0:
            window_length -= 1
        window_length = max(window_length, polyorder + 2)

        # Use polyorder=2 instead of 3 for lighter filtering
        filtered = savgol_filter(data[column].values, window_length, polyorder=2)
        data[f'{column}_Filtered'] = filtered

        noise_reduction = np.std(data[column] - filtered)
        print(f"✓ Applied Savitzky-Golay filter (window={window_length}, poly=2)")
        print(f"✓ Noise reduction (std): {noise_reduction:.2f}")

        return data

    def calculate_log_returns(self, data):
        """Calculate log returns for stationarity"""
        data = data.copy()
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
        data['Log_Return'] = data['Log_Return'].fillna(0)

        print(f"✓ Log returns calculated (mean: {data['Log_Return'].mean():.6f})")
        return data

    def calculate_technical_indicators(self, data):
        """Calculate technical indicators"""
        print("\n" + "=" * 80)
        print("STEP 3: CALCULATING TECHNICAL INDICATORS")
        print("=" * 80)

        data = data.copy()

        # Moving averages
        data['SMA_12'] = data['Close'].rolling(window=12).mean()
        data['SMA_24'] = data['Close'].rolling(window=24).mean()
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_24'] = data['Close'].ewm(span=24, adjust=False).mean()

        # RSI
        if len(data) >= 14:
            rsi = RSIIndicator(close=data['Close'], window=14)
            data['RSI'] = rsi.rsi()
        else:
            data['RSI'] = 50

        # Bollinger Bands
        if len(data) >= 20:
            bb = BollingerBands(close=data['Close'], window=20, window_dev=2)
            data['BB_High'] = bb.bollinger_hband()
            data['BB_Low'] = bb.bollinger_lband()
            data['BB_Mid'] = bb.bollinger_mavg()
        else:
            data['BB_High'] = data['Close']
            data['BB_Low'] = data['Close']
            data['BB_Mid'] = data['Close']

        # Volatility
        data['Volatility'] = data['Log_Return'].rolling(window=12).std()

        # Price momentum
        data['Momentum'] = data['Close'] - data['Close'].shift(12)

        print(f"✓ Technical indicators calculated")
        return data

    def resample_to_5min(self, data):
        """
        Since this is 15-minute data, we'll skip resampling
        and just use the same data for both branches
        """
        print("\n" + "=" * 80)
        print("STEP 4: HANDLING TIMEFRAME")
        print("=" * 80)

        # Detect current interval
        if len(data) > 1:
            interval = (data.index[1] - data.index[0]).total_seconds() / 60
            print(f"✓ Current interval: ~{interval:.1f} minutes")

        # For 15-minute data, we'll use the same data for both branches
        print(f"✓ Using 15-minute data for both branches")
        print(f"✓ Data shape: {data.shape}")

        return data.copy()

    def create_sliding_windows(self, data_1min, data_5min):
        """
        CRITICAL FIX: Custom train-test split (1500 train / 1444 test)
        """
        print("\n" + "=" * 80)
        print("STEP 5: CREATING SLIDING WINDOWS")
        print(f"TARGET: {self.train_samples} train / {self.test_samples} test")
        print("=" * 80)

        # Feature columns (exclude target)
        features = ['Log_Return', 'SMA_12', 'EMA_12', 'RSI', 'BB_Mid', 'Volatility', 'Momentum']

        # Remove rows with NaN
        data_1min = data_1min[features + ['Close']].dropna()
        data_5min = data_5min[features + ['Close']].dropna()

        print(f"✓ Features used: {features}")
        print(f"✓ 15-minute data after dropna: {data_1min.shape}")

        # CRITICAL: Custom split based on entry counts
        # We need to account for window_size when creating samples
        available_samples = len(data_5min) - self.window_size - 1
        print(f"✓ Available samples after windowing: {available_samples}")

        # CRITICAL: Dynamic split based on ratio if no fixed samples provided
        if self.train_samples is None:
            self.train_samples = int(available_samples * self.split_ratio)
            self.test_samples = available_samples - self.train_samples
            print(f"✓ Dynamic Split ({self.split_ratio*100:.0f}/{ (1-self.split_ratio)*100:.0f}):")
            print(f"  - Train samples: {self.train_samples}")
            print(f"  - Test samples: {self.test_samples}")
        elif available_samples < (self.train_samples + self.test_samples):
            print(f"⚠️  Warning: Not enough data for {self.train_samples + self.test_samples} samples")
            print(f"⚠️  Adjusting to available samples: {available_samples}")
            # Maintain ratio
            ratio = self.train_samples / (self.train_samples + self.test_samples)
            self.train_samples = int(available_samples * ratio)
            self.test_samples = available_samples - self.train_samples

        # Calculate split index
        split_idx = self.window_size + self.train_samples

        # Split data
        data_5min_train = data_5min.iloc[:split_idx]
        data_5min_test = data_5min.iloc[split_idx - self.window_size:]  # Overlap for continuity

        # Split 1-min data (same as 5-min since it's 15-minute data)
        data_1min_train = data_1min.iloc[:split_idx]
        data_1min_test = data_1min.iloc[split_idx - self.window_size:]

        print(f"\n✓ Data split:")
        print(f"  Train: {len(data_5min_train)} entries")
        print(f"  Test: {len(data_5min_test)} entries")

        # FIT scalers on TRAIN data only
        self.feature_scaler_1min.fit(data_1min_train[features])
        self.feature_scaler_5min.fit(data_5min_train[features])
        self.target_scaler.fit(data_5min_train[['Close']])

        # Transform train and test separately
        data_1min_train_scaled = pd.DataFrame(
            self.feature_scaler_1min.transform(data_1min_train[features]),
            index=data_1min_train.index,
            columns=features
        )
        data_1min_test_scaled = pd.DataFrame(
            self.feature_scaler_1min.transform(data_1min_test[features]),
            index=data_1min_test.index,
            columns=features
        )

        data_5min_train_scaled = pd.DataFrame(
            self.feature_scaler_5min.transform(data_5min_train[features]),
            index=data_5min_train.index,
            columns=features
        )
        data_5min_test_scaled = pd.DataFrame(
            self.feature_scaler_5min.transform(data_5min_test[features]),
            index=data_5min_test.index,
            columns=features
        )

        # Scale target prices
        data_5min_train['Close_Scaled'] = self.target_scaler.transform(data_5min_train[['Close']])
        data_5min_test['Close_Scaled'] = self.target_scaler.transform(data_5min_test[['Close']])

        print("✓ Scaling complete (NO data leakage!)")

        # Create windows for train
        X_1min_train, X_5min_train, y_train, y_direction_train = self._create_windows_helper(
            data_1min_train_scaled, data_5min_train_scaled, data_5min_train, 'TRAIN'
        )

        # Create windows for test
        X_1min_test, X_5min_test, y_test, y_direction_test = self._create_windows_helper(
            data_1min_test_scaled, data_5min_test_scaled, data_5min_test, 'TEST'
        )

        print(f"\n✅ FINAL SPLIT:")
        print(f"   Train samples: {len(y_train)}")
        print(f"   Test samples: {len(y_test)}")

        return (X_1min_train, X_1min_test, X_5min_train, X_5min_test,
                y_train, y_test, y_direction_train, y_direction_test)

    def _create_windows_helper(self, data_1min_scaled, data_5min_scaled, data_5min_raw, split_name):
        """Helper to create windows"""
        features = data_1min_scaled.columns.tolist()

        X_1min_list = []
        X_5min_list = []
        y_price_list = []
        y_direction_list = []

        for i in range(self.window_size, len(data_5min_scaled) - 1):
            # 5-min window (in this case, 15-minute window)
            window_5min = data_5min_scaled.iloc[i - self.window_size:i].values

            # Corresponding 1-min window (same as 5-min for 15-minute data)
            window_1min = data_1min_scaled.iloc[i - self.window_size:i].values

            # Ensure windows have correct length
            if len(window_1min) != self.window_size or len(window_5min) != self.window_size:
                continue

            # CRITICAL FIX: Target is SCALED close price
            target_price = data_5min_raw.iloc[i + 1]['Close_Scaled']
            current_price = data_5min_raw.iloc[i]['Close_Scaled']

            # Direction: 1 if up, 0 if down
            direction = 1 if target_price > current_price else 0

            X_1min_list.append(window_1min)
            X_5min_list.append(window_5min)
            y_price_list.append(target_price)
            y_direction_list.append(direction)

        X_1min = np.array(X_1min_list)
        X_5min = np.array(X_5min_list)
        y_price = np.array(y_price_list)
        y_direction = np.array(y_direction_list)

        print(f"\n✓ {split_name} Windows created:")
        print(f"  - X_1min shape: {X_1min.shape}")
        print(f"  - X_5min shape: {X_5min.shape}")
        print(f"  - y_price shape: {y_price.shape}")
        print(f"  - Direction: Up={np.sum(y_direction)}, Down={len(y_direction) - np.sum(y_direction)}")

        return X_1min, X_5min, y_price, y_direction

    def process_pipeline(self):
        """Complete preprocessing pipeline"""
        # Load data
        data_1min = self.load_and_clean_data()

        # Apply DSP filtering
        data_1min = self.apply_dsp_filtering(data_1min)

        # Calculate log returns
        data_1min = self.calculate_log_returns(data_1min)

        # Technical indicators
        data_1min = self.calculate_technical_indicators(data_1min)

        # Create 5-minute data (for 15-minute data, just copy)
        data_5min = self.resample_to_5min(data_1min)

        # Create sliding windows (with custom train-test split)
        result = self.create_sliding_windows(data_1min, data_5min)

        print("\n" + "=" * 80)
        print("✅ PREPROCESSING COMPLETE!")
        print("=" * 80)

        return result


def main():
    """Main execution"""
    print("=" * 80)
    print("BITCOIN PRICE PREDICTION - CUSTOM DATA PREPROCESSING")
    print("File: btc_15m_data_2018_to_2025.csv")
    print("15-minute interval data")
    print("=" * 80)

    # Check if file exists
    csv_file = 'btc_15m_data_2018_to_2025.csv'

    if not os.path.exists(csv_file):
        print(f"\n❌ File not found: {csv_file}")
        print("Please ensure the file is in the current directory.")
        return

    print(f"\n✓ Found file: {csv_file}")

    # Process data with dynamic split (80/20)
    processor = EnhancedBitcoinProcessor(
        csv_path=csv_file,
        window_size=60,
        train_samples=None,  # Use dynamic split
        test_samples=None,
        split_ratio=0.8
    )

    result = processor.process_pipeline()

    if result:
        (X_1min_train, X_1min_test, X_5min_train, X_5min_test,
         y_price_train, y_price_test, y_direction_train, y_direction_test) = result

        # Save processed data
        print("\n💾 Saving processed data...")
        np.save('X_1min_train.npy', X_1min_train)
        np.save('X_1min_test.npy', X_1min_test)
        np.save('X_5min_train.npy', X_5min_train)
        np.save('X_5min_test.npy', X_5min_test)
        np.save('y_price_train.npy', y_price_train)
        np.save('y_price_test.npy', y_price_test)
        np.save('y_direction_train.npy', y_direction_train)
        np.save('y_direction_test.npy', y_direction_test)

        # CRITICAL: Save scalers for inverse transform
        with open('target_scaler.pkl', 'wb') as f:
            pickle.dump(processor.target_scaler, f)

        print("✓ Files saved:")
        print("  - X_1min_train.npy, X_1min_test.npy")
        print("  - X_5min_train.npy, X_5min_test.npy")
        print("  - y_price_train.npy, y_price_test.npy")
        print("  - y_direction_train.npy, y_direction_test.npy")
        print("  - target_scaler.pkl (for inverse transform)")

        print(f"\n📊 FINAL STATISTICS:")
        print(f"   Training samples: {len(y_price_train)}")
        print(f"   Testing samples: {len(y_price_test)}")
        print(f"   Total samples: {len(y_price_train) + len(y_price_test)}")

        print("\n✨ Ready for model training! Run: python train_fixed.py")


if __name__ == "__main__":
    main()

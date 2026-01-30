import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np


def huber_loss(y_true, y_pred, delta=1.0):
    """Huber loss - more robust to outliers than MSE"""
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= delta
    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * (tf.abs(error) - 0.5 * delta)
    return tf.where(is_small_error, squared_loss, linear_loss)


def directional_loss(y_true, y_pred):
    """Custom loss that prioritizes getting the direction correct"""
    # Standard MSE component
    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    # Directional component
    true_direction = tf.sign(y_true[1:] - y_true[:-1])
    pred_direction = tf.sign(y_pred[1:] - y_pred[:-1])
    direction_error = tf.reduce_mean(tf.abs(true_direction - pred_direction))

    # Combined loss (70% direction, 30% MSE)
    return 0.3 * mse + 0.7 * direction_error


class CNNLSTMHybrid:
    """
    CNN-LSTM Hybrid Model for Bitcoin Price Prediction

    Architecture:
    1. Dual-path CNN for feature extraction (1-min and 5-min)
    2. LSTM for sequence learning
    3. Dense layers for final prediction
    """

    def __init__(self, window_size=60, n_features=8):
        self.window_size = window_size
        self.n_features = n_features
        self.model = None

    def build_cnn_branch(self, input_layer, name_prefix):
        """
        STAGE 3: FEATURE EXTRACTION (CNN)
        Detects short-term patterns using 1D convolutions

        WHERE CNN HAPPENS:
        - Conv1D layers extract local patterns from sequential data
        - Each Conv1D has filters (kernels) that slide across time dimension
        - MaxPooling reduces dimensionality while keeping important features
        """
        # First Conv Block - EXTRACTS SHORT-TERM PATTERNS
        x = layers.Conv1D(
            filters=64,
            kernel_size=3,
            padding='same',
            name=f'{name_prefix}_conv1'
        )(input_layer)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(pool_size=2, name=f'{name_prefix}_pool1')(x)
        x = layers.Dropout(0.2)(x)

        # Second Conv Block - EXTRACTS MID-TERM PATTERNS
        x = layers.Conv1D(
            filters=128,
            kernel_size=3,
            padding='same',
            name=f'{name_prefix}_conv2'
        )(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling1D(pool_size=2, name=f'{name_prefix}_pool2')(x)
        x = layers.Dropout(0.2)(x)

        # Third Conv Block - REFINES FEATURES
        x = layers.Conv1D(
            filters=64,
            kernel_size=3,
            padding='same',
            name=f'{name_prefix}_conv3'
        )(x)
        x = layers.BatchNormalization(name=f'{name_prefix}_bn3')(x)
        x = layers.Activation('relu')(x)

        return x

    def build_model(self):
        """
        Build the complete CNN-LSTM Hybrid model

        Flow:
        [1-min Data] → CNN → ┐
                              ├→ Concatenate → LSTM → Dense → Prediction
        [5-min Data] → CNN → ┘

        EXPLANATION:
        - TWO separate inputs: 1-min and 5-min timeframes
        - Each input goes through CNN (feature extraction)
        - CNN outputs are concatenated (combined)
        - Combined features go to LSTM (sequence learning)
        - LSTM output goes to Dense layers (final prediction)
        """
        print("\n" + "=" * 80)
        print("BUILDING CNN-LSTM HYBRID MODEL")
        print("=" * 80)

        # ===== INPUT LAYERS =====
        input_1min = layers.Input(
            shape=(self.window_size, self.n_features),
            name='input_1min'
        )
        input_5min = layers.Input(
            shape=(self.window_size, self.n_features),
            name='input_5min'
        )

        print("✓ Input layers created")
        print(f"  - 1-min input: {(self.window_size, self.n_features)}")
        print(f"  - 5-min input: {(self.window_size, self.n_features)}")

        # ===== BRANCH A: 1-MIN CNN (The Microscope) =====
        print("\n✓ Building Branch A: 1-Minute CNN (Micro-patterns)")
        cnn_1min = self.build_cnn_branch(input_1min, 'branch_1min')

        # ===== BRANCH B: 5-MIN CNN (The Map) =====
        print("✓ Building Branch B: 5-Minute CNN (Macro-trends)")
        cnn_5min = self.build_cnn_branch(input_5min, 'branch_5min')

        # ===== CONCATENATE CNN OUTPUTS =====
        print("\n✓ Concatenating CNN features")
        concatenated = layers.Concatenate(axis=-1, name='concatenate_cnn')([cnn_1min, cnn_5min])

        # ===== STAGE 4: MEMORY & TREND (LSTM) =====
        print("✓ Building LSTM layers (Sequence learning)")
        print("  WHERE LSTM HAPPENS: Below layers learn temporal dependencies")

        # First LSTM layer - LEARNS LONG-TERM DEPENDENCIES
        lstm1 = layers.LSTM(
            units=128,
            return_sequences=True,  # Returns full sequence for next LSTM
            dropout=0.3,
            recurrent_dropout=0.2,
            name='lstm1'
        )(concatenated)
        lstm1 = layers.BatchNormalization(name='lstm1_bn')(lstm1)

        # Second LSTM layer - LEARNS HIGHER-LEVEL PATTERNS
        lstm2 = layers.LSTM(
            units=64,
            return_sequences=False,  # Returns only final output
            dropout=0.3,
            recurrent_dropout=0.2,
            name='lstm2'
        )(lstm1)
        lstm2 = layers.BatchNormalization(name='lstm2_bn')(lstm2)

        # ===== STAGE 5: OUTPUT & OPTIMIZATION =====
        print("✓ Building output layers")

        # Dense layers
        dense1 = layers.Dense(64, activation='relu', name='dense1')(lstm2)
        dense1 = layers.Dropout(0.3)(dense1)

        dense2 = layers.Dense(32, activation='relu', name='dense2')(dense1)
        dense2 = layers.Dropout(0.2)(dense2)

        # Output layer (price prediction)
        output = layers.Dense(1, activation='linear', name='output_price')(dense2)

        # ===== CREATE MODEL =====
        self.model = Model(
            inputs=[input_1min, input_5min],
            outputs=output,
            name='CNN_LSTM_Hybrid'
        )

        print("\n" + "=" * 80)
        print("✅ MODEL ARCHITECTURE COMPLETE")
        print("=" * 80)

        return self.model

    def compile_model(self, learning_rate=0.001, loss_type='huber'):
        """
        Compile the model with specified loss function

        Args:
            learning_rate: Learning rate for Adam optimizer
            loss_type: 'huber', 'mse', or 'directional'
        """
        if loss_type == 'huber':
            loss = lambda y_true, y_pred: huber_loss(y_true, y_pred, delta=1.0)
        elif loss_type == 'directional':
            loss = directional_loss
        else:
            loss = 'mse'

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=['mae', 'mse']
        )

        print(f"\n✓ Model compiled with {loss_type.upper()} loss")
        print(f"✓ Learning rate: {learning_rate}")

    def summary(self):
        """Print model summary"""
        print("\n" + "=" * 80)
        print("MODEL SUMMARY")
        print("=" * 80)
        self.model.summary()

        # Count parameters
        trainable = np.sum([np.prod(v.shape) for v in self.model.trainable_weights])
        non_trainable = np.sum([np.prod(v.shape) for v in self.model.non_trainable_weights])

        print(f"\n✓ Total parameters: {trainable + non_trainable:,}")
        print(f"✓ Trainable parameters: {trainable:,}")
        print(f"✓ Non-trainable parameters: {non_trainable:,}")


class DirectionalAccuracyCallback(keras.callbacks.Callback):
    """
    Custom callback to track directional accuracy during training
    FIXED: Now handles scaler for inverse transform
    """
    def __init__(self, validation_data, scaler=None):
        super().__init__()
        self.validation_data = validation_data
        self.scaler = scaler

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_pred = self.model.predict(X_val, verbose=0).flatten()

        # Inverse transform if scaler provided
        if self.scaler is not None:
            y_val = self.scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
            y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

        # Calculate directional accuracy
        y_val_direction = np.diff(y_val) > 0
        y_pred_direction = np.diff(y_pred) > 0

        if len(y_val_direction) > 0:
            directional_accuracy = np.mean(y_val_direction == y_pred_direction) * 100
            logs['val_directional_accuracy'] = directional_accuracy

            if epoch % 5 == 0:
                print(f"\n  Directional Accuracy: {directional_accuracy:.2f}%")


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("CNN-LSTM HYBRID MODEL - ARCHITECTURE TEST")
    print("=" * 80)

    # Create model
    model = CNNLSTMHybrid(window_size=60, n_features=7)  # 7 features
    model.build_model()
    model.compile_model(learning_rate=0.001, loss_type='mse')
    model.summary()

    print("\n✅ Model architecture verified!")
    print("Ready for training with real data.")

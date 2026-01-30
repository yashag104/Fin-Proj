import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from model import CNNLSTMHybrid, DirectionalAccuracyCallback
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive evaluation metrics"""
    # RMSE (Root Mean Squared Error)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(y_true - y_pred))

    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    # Directional Accuracy
    y_true_direction = np.diff(y_true) > 0
    y_pred_direction = np.diff(y_pred) > 0
    directional_accuracy = np.mean(y_true_direction == y_pred_direction) * 100

    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'Directional_Accuracy': directional_accuracy
    }


def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    """Calculate Sharpe Ratio"""
    excess_returns = returns - risk_free_rate
    if np.std(returns) == 0:
        return 0
    sharpe = np.mean(excess_returns) / np.std(returns)
    return sharpe * np.sqrt(252)  # Annualized


def calculate_max_drawdown(prices):
    """Calculate Maximum Drawdown"""
    cumulative = np.maximum.accumulate(prices)
    drawdown = (prices - cumulative) / cumulative
    max_drawdown = np.min(drawdown) * 100
    return max_drawdown


def walk_forward_validation(model, X_1min_test, X_5min_test, y_test, window_size=10):
    """Walk-forward validation"""
    print("\n" + "=" * 80)
    print("WALK-FORWARD VALIDATION")
    print("=" * 80)

    predictions = []

    for i in range(0, len(y_test), window_size):
        end_idx = min(i + window_size, len(y_test))

        X_1min_batch = X_1min_test[i:end_idx]
        X_5min_batch = X_5min_test[i:end_idx]

        pred = model.predict([X_1min_batch, X_5min_batch], verbose=0)
        predictions.extend(pred.flatten())

        if (i // window_size) % 10 == 0:
            progress = (end_idx / len(y_test)) * 100
            print(f"Progress: {progress:.1f}%")

    predictions = np.array(predictions[:len(y_test)])

    print("✓ Walk-forward validation complete")
    return predictions


def plot_training_history(history, save_path='training_history_fixed.png'):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss
    axes[0, 0].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # MAE
    axes[0, 1].plot(history.history['mae'], label='Train MAE', linewidth=2)
    axes[0, 1].plot(history.history['val_mae'], label='Val MAE', linewidth=2)
    axes[0, 1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # MSE
    axes[1, 0].plot(history.history['mse'], label='Train MSE', linewidth=2)
    axes[1, 0].plot(history.history['val_mse'], label='Val MSE', linewidth=2)
    axes[1, 0].set_title('Mean Squared Error', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MSE')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Directional Accuracy (if available)
    if 'val_directional_accuracy' in history.history:
        axes[1, 1].plot(history.history['val_directional_accuracy'],
                       label='Directional Accuracy', linewidth=2, color='green')
        axes[1, 1].set_title('Validation Directional Accuracy', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Directional Accuracy\nNot Tracked',
                       ha='center', va='center', fontsize=12)
        axes[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Training history saved to {save_path}")
    plt.close()


def plot_predictions(y_true, y_pred, save_path='predictions_fixed.png', num_samples=None):
    """Plot actual vs predicted prices"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Use all samples if num_samples not specified
    if num_samples is None:
        num_samples = len(y_true)

    plot_len = min(num_samples, len(y_true))
    x_axis = np.arange(plot_len)

    # Plot 1: Actual vs Predicted
    axes[0].plot(x_axis, y_true[:plot_len], 'b-', linewidth=2,
                label='Actual Price', alpha=0.8)
    axes[0].plot(x_axis, y_pred[:plot_len], 'r--', linewidth=2,
                label='Predicted Price', alpha=0.8)
    axes[0].set_title('Actual vs Predicted Bitcoin Prices (FIXED)',
                     fontsize=15, fontweight='bold')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Price (USD)')
    axes[0].legend(loc='best', fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Prediction Error
    error = y_pred[:plot_len] - y_true[:plot_len]
    error_pct = (error / y_true[:plot_len]) * 100
    axes[1].plot(x_axis, error_pct, 'purple', linewidth=1.5, alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].fill_between(x_axis, error_pct, 0, where=(error_pct > 0),
                        color='green', alpha=0.3, label='Overestimate')
    axes[1].fill_between(x_axis, error_pct, 0, where=(error_pct < 0),
                        color='red', alpha=0.3, label='Underestimate')
    axes[1].set_title('Prediction Error (%)', fontsize=15, fontweight='bold')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Error (%)')
    axes[1].legend(loc='best', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Predictions plot saved to {save_path}")
    plt.close()


def plot_directional_accuracy(y_true, y_pred, save_path='directional_analysis_fixed.png'):
    """Plot directional accuracy analysis"""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Calculate directions
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    correct = true_direction == pred_direction

    x_axis = np.arange(len(true_direction))

    # Plot 1: Direction comparison
    axes[0].scatter(x_axis[correct], true_direction[correct],
                   c='green', marker='o', s=30, alpha=0.6, label='Correct')
    axes[0].scatter(x_axis[~correct], true_direction[~correct],
                   c='red', marker='x', s=30, alpha=0.6, label='Incorrect')
    axes[0].set_title('Direction Prediction Accuracy (Green=Correct, Red=Wrong)',
                     fontsize=15, fontweight='bold')
    axes[0].set_xlabel('Time Steps')
    axes[0].set_ylabel('Direction (0=Down, 1=Up)')
    axes[0].legend(loc='best', fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Rolling accuracy
    window = min(50, len(correct) // 10)  # Adaptive window size
    if window < 5:
        window = 5

    rolling_accuracy = np.convolve(correct.astype(float),
                                   np.ones(window)/window, mode='valid')
    axes[1].plot(rolling_accuracy * 100, linewidth=2, color='blue')
    axes[1].axhline(y=50, color='red', linestyle='--',
                   linewidth=2, label='Random Baseline (50%)')
    axes[1].fill_between(range(len(rolling_accuracy)), 50, rolling_accuracy * 100,
                        where=(rolling_accuracy * 100 > 50),
                        color='green', alpha=0.3, label='Above Baseline')
    axes[1].set_title(f'Rolling Directional Accuracy (Window={window})',
                     fontsize=15, fontweight='bold')
    axes[1].set_xlabel('Time Steps')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend(loc='best', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Directional analysis saved to {save_path}")
    plt.close()


def main():
    """Main training script - FIXED VERSION"""
    print("=" * 80)
    print("BITCOIN PRICE PREDICTION - CNN-LSTM HYBRID TRAINING (FIXED)")
    print("=" * 80)

    # Load processed data
    print("\n📂 Loading processed data...")
    try:
        X_1min_train = np.load('X_1min_train.npy')
        X_1min_test = np.load('X_1min_test.npy')
        X_5min_train = np.load('X_5min_train.npy')
        X_5min_test = np.load('X_5min_test.npy')
        y_price_train = np.load('y_price_train.npy')
        y_price_test = np.load('y_price_test.npy')

        # CRITICAL: Load scaler for inverse transform
        with open('target_scaler.pkl', 'rb') as f:
            target_scaler = pickle.load(f)

        print("✓ Data loaded successfully!")
        print(f"  Train samples: {len(y_price_train)}")
        print(f"  Test samples: {len(y_price_test)}")
        print("✓ Target scaler loaded for inverse transform")
    except FileNotFoundError:
        print("\n❌ Processed data not found!")
        print("Please run: python data_preprocessing_fixed.py first")
        return

    # Build model
    print("\n🏗️  Building model...")
    window_size = X_1min_train.shape[1]
    n_features = X_1min_train.shape[2]

    model_builder = CNNLSTMHybrid(window_size=window_size, n_features=n_features)
    model_builder.build_model()

    # CRITICAL: Use MSE loss instead of Huber for better price regression
    model_builder.compile_model(learning_rate=0.0005, loss_type='mse')
    model_builder.summary()

    # Training configuration
    print("\n⚙️  Training configuration:")
    EPOCHS = 150  # Increased for better convergence
    BATCH_SIZE = 32  # Increased batch size
    VALIDATION_SPLIT = 0.1

    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Validation split: {VALIDATION_SPLIT}")

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,  # Increased patience
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,  # Increased patience
            min_lr=1e-7,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        DirectionalAccuracyCallback(
            validation_data=([X_1min_test, X_5min_test], y_price_test),
            scaler=target_scaler  # Pass scaler to callback
        )
    ]

    # Train model
    print("\n🚀 Starting training...")
    print("=" * 80)

    history = model_builder.model.fit(
        [X_1min_train, X_5min_train],
        y_price_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=callbacks,
        verbose=1
    )

    print("\n✅ Training complete!")

    # Save final model
    model_builder.model.save('final_model.keras')
    print("\n💾 Model saved: final_model.keras, best_model.keras")

    # Plot training history
    plot_training_history(history)

    # Evaluate on test set
    print("\n" + "=" * 80)
    print("EVALUATION ON TEST SET")
    print("=" * 80)

    # CRITICAL: Predict in scaled space, then inverse transform
    y_pred_test_scaled = model_builder.model.predict(
        [X_1min_test, X_5min_test]
    ).flatten()

    # Inverse transform to get actual prices
    y_pred_test = target_scaler.inverse_transform(
        y_pred_test_scaled.reshape(-1, 1)
    ).flatten()

    y_test_actual = target_scaler.inverse_transform(
        y_price_test.reshape(-1, 1)
    ).flatten()

    print("✓ Predictions inverse-transformed to actual prices")

    # Calculate metrics
    metrics = calculate_metrics(y_test_actual, y_pred_test)

    print("\n📊 Performance Metrics:")
    print(f"  RMSE: ${metrics['RMSE']:.2f}")
    print(f"  MAE: ${metrics['MAE']:.2f}")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    print(f"  Directional Accuracy: {metrics['Directional_Accuracy']:.2f}%")

    # Calculate returns and Sharpe Ratio
    returns_actual = np.diff(y_test_actual) / y_test_actual[:-1]
    returns_pred = np.diff(y_pred_test) / y_pred_test[:-1]

    sharpe_actual = calculate_sharpe_ratio(returns_actual)
    sharpe_pred = calculate_sharpe_ratio(returns_pred)

    print(f"\n📈 Trading Metrics:")
    print(f"  Sharpe Ratio (Actual): {sharpe_actual:.4f}")
    print(f"  Sharpe Ratio (Predicted): {sharpe_pred:.4f}")

    # Maximum Drawdown
    max_dd_actual = calculate_max_drawdown(y_test_actual)
    max_dd_pred = calculate_max_drawdown(y_pred_test)

    print(f"  Maximum Drawdown (Actual): {max_dd_actual:.2f}%")
    print(f"  Maximum Drawdown (Predicted): {max_dd_pred:.2f}%")

    # Walk-forward validation
    y_pred_walk_scaled = walk_forward_validation(
        model_builder.model, X_1min_test, X_5min_test, y_price_test
    )

    y_pred_walk = target_scaler.inverse_transform(
        y_pred_walk_scaled.reshape(-1, 1)
    ).flatten()

    metrics_walk = calculate_metrics(y_test_actual, y_pred_walk)
    print("\n📊 Walk-Forward Validation Metrics:")
    print(f"  RMSE: ${metrics_walk['RMSE']:.2f}")
    print(f"  Directional Accuracy: {metrics_walk['Directional_Accuracy']:.2f}%")

    # Generate plots
    print("\n📊 Generating visualizations...")
    plot_predictions(y_test_actual, y_pred_test)
    plot_directional_accuracy(y_test_actual, y_pred_test)

    # Save results
    results = {
        'RMSE': metrics['RMSE'],
        'MAE': metrics['MAE'],
        'MAPE': metrics['MAPE'],
        'Directional_Accuracy': metrics['Directional_Accuracy'],
        'Sharpe_Ratio_Actual': sharpe_actual,
        'Sharpe_Ratio_Predicted': sharpe_pred,
        'Max_Drawdown_Actual': max_dd_actual,
        'Max_Drawdown_Predicted': max_dd_pred,
        'Walk_Forward_RMSE': metrics_walk['RMSE'],
        'Walk_Forward_DA': metrics_walk['Directional_Accuracy']
    }

    np.save('evaluation_results_fixed.npy', results)

    # Print final report
    print("\n" + "=" * 80)
    print("✅ TRAINING AND EVALUATION COMPLETE!")
    print("=" * 80)
    print("\n📁 Generated Files:")
    print("  ✓ best_model.keras - Best model during training")
    print("  ✓ final_model.keras - Final trained model")
    print("  ✓ training_history_fixed.png - Training curves")
    print("  ✓ predictions_fixed.png - Actual vs Predicted")
    print("  ✓ directional_analysis_fixed.png - Direction accuracy")
    print("  ✓ evaluation_results_fixed.npy - Metrics dictionary")

    print("\n🎯 Key Takeaways:")
    if metrics['Directional_Accuracy'] > 55:
        print(f"  ✅ Strong directional accuracy: {metrics['Directional_Accuracy']:.2f}%")
    else:
        print(f"  ⚠️  Directional accuracy needs improvement: {metrics['Directional_Accuracy']:.2f}%")

    if sharpe_pred > 0:
        print(f"  ✅ Positive Sharpe ratio: {sharpe_pred:.4f}")
    else:
        print(f"  ⚠️  Negative Sharpe ratio: {sharpe_pred:.4f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

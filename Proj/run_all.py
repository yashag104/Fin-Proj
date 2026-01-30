"""
Quick Start Script - Runs the entire pipeline
"""
import os
import sys
import subprocess


def run_command(command, description):
    """Run a command and handle errors"""
    print("\n" + "=" * 80)
    print(f"🚀 {description}")
    print("=" * 80)

    try:
        result = subprocess.run(
            [sys.executable, command],
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✅ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FAILED")
        print(f"Error: {e}")
        return False


def main():
    print("=" * 80)
    print("BITCOIN PRICE PREDICTION - COMPLETE PIPELINE")
    print("CNN-LSTM Hybrid Model")
    print("=" * 80)

    # Check if CSV exists
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if not csv_files:
        print("\n❌ ERROR: No CSV file found!")
        print("\n📋 Please place your Bitcoin CSV file in this directory first.")
        print("The CSV should contain columns like: Open time, Open, High, Low, Close, Volume")
        return

    print(f"\n✓ Found CSV file: {csv_files[0]}")

    # Step 1: Data Preprocessing
    if not run_command('data_preprocessing.py', 'Step 1: Data Preprocessing'):
        return

    # Step 2: Model Training
    if not run_command('train.py', 'Step 2: Model Training & Evaluation'):
        return

    # Success
    print("\n" + "=" * 80)
    print("✅ COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
    print("=" * 80)

    print("\n📁 Generated Files:")
    files_to_check = [
        'best_model.keras',
        'final_model.keras',
        'training_history.png',
        'predictions.png',
        'directional_analysis.png'
    ]

    for file in files_to_check:
        if os.path.exists(file):
            print(f"  ✓ {file}")

    print("\n🎯 Next Steps:")
    print("  1. Review training_history.png for training progress")
    print("  2. Check predictions.png for model accuracy")
    print("  3. Analyze directional_analysis.png for trading potential")
    print("  4. Load best_model.keras for making predictions")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

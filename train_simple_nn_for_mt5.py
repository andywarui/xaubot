"""
Train a simple feedforward neural network for MT5 compatibility.
Uses only basic ONNX operators that MT5 supports.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

# Simple feedforward network
class SimpleClassifier(nn.Module):
    def __init__(self, input_size=26, hidden_size=64, num_classes=3):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

def train_nn():
    print("=" * 70)
    print("TRAINING SIMPLE NEURAL NETWORK FOR MT5")
    print("=" * 70)

    # Load the same data used for LightGBM
    data_path = Path("data/processed/labeled_overlap_M1.parquet")

    if not data_path.exists():
        print(f"ERROR: Data not found at {data_path}")
        print("Run: python src/create_labels.py")
        return False

    print(f"\nLoading data from: {data_path}")
    df = pd.read_parquet(data_path)

    # Use same features as LightGBM
    feature_cols = [
        'body', 'body_abs', 'candle_range', 'close_position',
        'return_1', 'return_5', 'return_15', 'return_60',
        'tr', 'atr_14', 'rsi_14', 'ema_10', 'ema_20', 'ema_50',
        'hour_sin', 'hour_cos',
        'M5_trend', 'M5_position',
        'M15_trend', 'M15_position',
        'H1_trend', 'H1_position',
        'H4_trend', 'H4_position',
        'D1_trend', 'D1_position'
    ]

    X = df[feature_cols].values.astype(np.float32)
    y = df['label'].values.astype(np.int64)

    print(f"  Samples: {len(X):,}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Classes: {len(np.unique(y))}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")

    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)

    # Create model
    model = SimpleClassifier(input_size=26, hidden_size=64, num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training
    print("\nTraining...")
    batch_size = 1024
    epochs = 50

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        # Mini-batch training
        for i in range(0, len(X_train_t), batch_size):
            batch_X = X_train_t[i:i+batch_size]
            batch_y = y_train_t[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_t)
                _, predicted = torch.max(test_outputs, 1)
                acc = accuracy_score(y_test_t.numpy(), predicted.numpy())
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(X_train_t):.4f} | Test Acc: {acc:.4f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        train_outputs = model(X_train_t)
        _, train_pred = torch.max(train_outputs, 1)
        train_acc = accuracy_score(y_train_t.numpy(), train_pred.numpy())

        test_outputs = model(X_test_t)
        _, test_pred = torch.max(test_outputs, 1)
        test_acc = accuracy_score(y_test_t.numpy(), test_pred.numpy())

    print(f"\nFinal Results:")
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")

    # Export to ONNX
    print("\nExporting to ONNX...")
    model.eval()

    dummy_input = torch.randn(1, 26)
    onnx_path = "python_training/models/simple_nn_mt5.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"  Saved: {onnx_path}")

    # Validate ONNX
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX model is valid")

    # Test ONNX inference
    sess = ort.InferenceSession(onnx_path)
    test_input = X_test[:5].astype(np.float32)
    onnx_outputs = sess.run(None, {'input': test_input})

    print("\nONNX Test (first 5 samples):")
    print("  Predictions:", onnx_outputs[0])
    print("  Shape:", onnx_outputs[0].shape)

    # Copy to MT5 folders
    import shutil
    destinations = [
        "mt5_expert_advisor/Files/lightgbm_xauusd.onnx",  # Replace LightGBM
        "MT5_XAUBOT/Files/lightgbm_xauusd.onnx"
    ]

    for dest in destinations:
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(onnx_path, dest)
        print(f"  Copied to: {dest}")

    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nModel: {onnx_path}")
    print(f"Accuracy: {test_acc:.2%}")
    print("\nNext: Copy to MT5 and test!")

    return True

if __name__ == "__main__":
    success = train_nn()
    exit(0 if success else 1)

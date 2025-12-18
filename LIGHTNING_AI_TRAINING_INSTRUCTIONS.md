# Lightning.ai Training Instructions for Code Agent

## Objective
Train a Multi-Timeframe Transformer model for XAUUSD price prediction and integrate it with LightGBM classifier to improve trading performance from 75.88% win rate to >78%.

## System Requirements
- Python 3.10+
- GPU recommended (T4/V100/A100 for faster training)
- ~2GB disk space for data
- ~500MB for model checkpoints

---

## Phase 1: Environment Setup

### Step 1: Clone Repository and Setup Git LFS
```bash
# Clone the repository
git clone https://github.com/andywarui/xaubot
cd xaubot

# Install and configure Git LFS
git lfs install

# Pull all large files (parquet/csv data - ~1.6 GB)
git lfs pull

# Verify data files exist
ls -lh data/processed/features_*.parquet
ls -lh data/processed/xauusd_*.parquet
```

**Alternative: Generate Synthetic Data for Testing (Optional)**

If you want to test the pipeline without downloading the full dataset (for development/testing only):

```bash
# Install dependencies first
pip install pandas numpy pyarrow

# Generate synthetic XAUUSD data (~5 MB instead of 1.6 GB)
python generate_synthetic_data.py

# Then generate features from synthetic data
python python_training/build_features_all_tf.py

# ⚠️ WARNING: Synthetic data is for TESTING ONLY
# Production models MUST use real market data (git lfs pull)
```

**Expected Output:** 26 files in `data/processed/` including:
- `features_m1_train.parquet`, `features_m1_val.parquet`, `features_m1_test.parquet`
- `features_m5_train.parquet`, `features_m5_val.parquet`, `features_m5_test.parquet`
- `features_m15_train.parquet`, `features_m15_val.parquet`, `features_m15_test.parquet`
- `features_h1_train.parquet`, `features_h1_val.parquet`, `features_h1_test.parquet`
- `features_d1_train.parquet`, `features_d1_val.parquet`, `features_d1_test.parquet`
- `xauusd_M1.parquet`, `xauusd_M5.parquet`, `xauusd_M15.parquet`, `xauusd_H1.parquet`, `xauusd_D1.parquet`
- Plus 7 timeframe parquets and 4 CSV files

### Step 2: Install Dependencies
```bash
# Install PyTorch and ML dependencies
pip install -r requirements_transformer.txt

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

**Expected Output:**
```
PyTorch: 2.x.x
CUDA Available: True  # (or False if CPU-only)
```

### Step 3: Verify Data Integrity
```bash
# Check feature dimensions and row counts
python -c "
import pandas as pd

# Check M1 features (primary timeframe)
m1_train = pd.read_parquet('data/processed/features_m1_train.parquet')
m1_val = pd.read_parquet('data/processed/features_m1_val.parquet')
m1_test = pd.read_parquet('data/processed/features_m1_test.parquet')

print(f'M1 Train: {m1_train.shape} (rows x features)')
print(f'M1 Val: {m1_val.shape}')
print(f'M1 Test: {m1_test.shape}')
print(f'Expected columns: 26 features + label + time')
print(f'Actual columns: {m1_train.columns.tolist()[:5]}... (showing first 5)')

# Check price data
prices = pd.read_parquet('data/processed/xauusd_M1.parquet')
print(f'\\nPrice data (M1): {prices.shape}')
print(f'Columns: {prices.columns.tolist()}')
"
```

**Expected Output:**
- M1 Train: ~600,000+ rows × 27-28 columns
- Features include: rsi_14, atr_14, bb_width, macd, etc.
- Price data has columns: time, open, high, low, close, volume

---

## Phase 2: Train Multi-Timeframe Transformer (Stage 1)

### Step 4: Train Transformer Model
```bash
# Run multi-timeframe transformer training
# This uses 130 features (26 features × 5 timeframes: M1, M5, M15, H1, D1)
python python_training/train_multi_tf_transformer.py

# Training will take 30-60 minutes depending on GPU
# Monitor output for:
# - Loss decreasing each epoch
# - Direction accuracy improving (target: 75-85%)
# - Early stopping after 5 epochs without improvement
```

**Expected Output During Training:**
```
Loading features for timeframes: m1, m5, m15, h1, d1
Train samples: ~420,000 | Val samples: ~60,000
Feature columns: 130 (26 features × 5 timeframes)

Epoch 1/30:
Train Loss: 0.0045 | Train Dir Acc: 58.3%
Val Loss: 0.0042 | Val Dir Acc: 59.1%

Epoch 10/30:
Train Loss: 0.0028 | Train Dir Acc: 71.2%
Val Loss: 0.0031 | Val Dir Acc: 69.8%

Epoch 25/30:
Train Loss: 0.0019 | Train Dir Acc: 78.4%
Val Loss: 0.0023 | Val Dir Acc: 76.2%

Early stopping triggered. Best Val Loss: 0.0023 at Epoch 25

Model saved to: python_training/models/multi_tf_transformer_price.pth
Scaler saved to: python_training/models/multi_tf_scaler.pkl
Config saved to: python_training/models/multi_tf_config.json
```

**Success Criteria:**
- ✅ Final validation direction accuracy: **75-85%**
- ✅ Val loss < 0.0025
- ✅ No overfitting (train acc - val acc < 10%)
- ✅ Files created: `multi_tf_transformer_price.pth`, `multi_tf_scaler.pkl`, `multi_tf_config.json`

### Step 5: Validate Transformer Output
```bash
# Test the trained transformer on validation data
python -c "
import torch
import pickle
import pandas as pd
import json
from python_training.train_multi_tf_transformer import MultiTFTransformer, load_multi_tf_split

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('python_training/models/multi_tf_config.json', 'r') as f:
    config = json.load(f)

model = MultiTFTransformer(
    input_dim=config['input_dim'],
    d_model=config['d_model'],
    n_heads=config['n_heads'],
    n_layers=config['n_layers'],
    dropout=config['dropout']
)
model.load_state_dict(torch.load('python_training/models/multi_tf_transformer_price.pth', map_location=device))
model.to(device)
model.eval()

# Load scaler
with open('python_training/models/multi_tf_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

print('✅ Transformer model loaded successfully')
print(f'Input dimension: {config[\"input_dim\"]} features')
print(f'Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M')
print(f'Device: {device}')
"
```

---

## Phase 3: Generate Hybrid Features (Stage 2)

### Step 6: Create Transformer Signal Feature
```bash
# Generate transformer predictions as feature #27
# This adds 'multi_tf_signal' column to M1 features
python python_training/prepare_hybrid_features_multi_tf.py

# Processing time: 5-10 minutes
# This will create 30-bar sequences, run inference, merge predictions back to M1 timeframe
```

**Expected Output:**
```
Loading trained Multi-TF Transformer model...
✅ Model loaded successfully

Loading price data for all timeframes (M1, M5, M15, H1, D1)...
✅ Price data loaded

Loading M1 feature splits...
Train: 420,000 rows | Val: 60,000 rows | Test: 60,000 rows

Generating predictions...
Processing: 100%|████████████| 13125/13125 [05:23<00:00, 40.61batch/s]

Adding multi_tf_signal to M1 features...
✅ Signal added as feature #27

Saving hybrid features...
✅ Saved: data/processed/features_m1_hybrid_train.parquet
✅ Saved: data/processed/features_m1_hybrid_val.parquet
✅ Saved: data/processed/features_m1_hybrid_test.parquet

Feature columns now: 27 (26 original + multi_tf_signal)
```

### Step 7: Verify Hybrid Features
```bash
python -c "
import pandas as pd

hybrid_train = pd.read_parquet('data/processed/features_m1_hybrid_train.parquet')
print(f'Hybrid feature shape: {hybrid_train.shape}')
print(f'Columns: {hybrid_train.columns.tolist()}')
print(f'\\nmulti_tf_signal stats:')
print(hybrid_train['multi_tf_signal'].describe())
print(f'\\nNull values: {hybrid_train[\"multi_tf_signal\"].isna().sum()}')
"
```

**Expected Output:**
- Shape: ~420,000 × 27
- `multi_tf_signal` present in columns
- Signal range: approximately -0.01 to +0.01 (normalized price change predictions)
- No null values

---

## Phase 4: Train Hybrid LightGBM (Stage 3)

### Step 8: Train Final Hybrid Model
```bash
# Train LightGBM classifier with 27 features (26 + transformer signal)
python python_training/train_lightgbm_hybrid.py

# Training time: 10-20 minutes
# This trains on 420k samples with cross-validation
```

**Expected Output:**
```
Loading hybrid features with transformer signal (27 features)...
Train: 420,000 | Val: 60,000 | Test: 60,000

Training LightGBM with 27 features...
[LightGBM] [Info] Number of positive: 140237, number of negative: 139763
[LightGBM] [Info] Total Bins: 4095
[LightGBM] [Info] Number of data points: 420000
[LightGBM] [Info] Number of features: 27

Training... Iteration 100: Train loss = 0.8432
Training... Iteration 200: Train loss = 0.7891
Training... Iteration 500: Train loss = 0.7234

✅ Training complete!

================== VALIDATION RESULTS ==================
Accuracy: 76.84%
Win Rate (Long): 77.21%
Win Rate (Short): 76.48%

Precision (Long): 0.782  |  Recall (Long): 0.791
Precision (Short): 0.754  |  Recall (Short): 0.738

Confusion Matrix:
              Predicted LONG  Predicted SHORT  Predicted NEUTRAL
Actual LONG         15,842         2,183             975
Actual SHORT         2,456        14,891           1,653
Actual NEUTRAL       3,124         2,987          15,889

================== COMPARISON WITH ORIGINAL MODEL ==================
                    Original     Hybrid      Improvement
Accuracy             75.88%      76.84%      +0.96%
Win Rate (Long)      76.12%      77.21%      +1.09%
Win Rate (Short)     75.64%      76.48%      +0.84%

Feature Importance (Top 10):
1. multi_tf_signal: 0.189  ⭐ NEW TRANSFORMER FEATURE
2. rsi_14: 0.092
3. atr_14: 0.078
4. macd: 0.064
5. bb_width: 0.058
...

Model saved: python_training/models/lightgbm_xauusd_hybrid.pkl
Metadata saved: python_training/models/model_metadata_hybrid.json
```

**Success Criteria:**
- ✅ Validation accuracy: **>76.5%** (improvement over 75.88% baseline)
- ✅ Win rate (Long): **>77%**
- ✅ Win rate (Short): **>76%**
- ✅ `multi_tf_signal` in top 5 feature importance
- ✅ Files created: `lightgbm_xauusd_hybrid.pkl`, `model_metadata_hybrid.json`

---

## Phase 5: Model Export and Validation (Optional)

### Step 9: Export to ONNX (if needed for MT5)
```bash
# Export hybrid LightGBM to ONNX format for MetaTrader 5
python python_training/export_to_onnx.py

# This converts the hybrid model to ONNX format for production deployment
```

### Step 10: Save Training Artifacts
```bash
# Archive all trained models and configs
mkdir -p trained_models_$(date +%Y%m%d)

# Copy transformer models
cp python_training/models/multi_tf_transformer_price.pth trained_models_*/
cp python_training/models/multi_tf_scaler.pkl trained_models_*/
cp python_training/models/multi_tf_config.json trained_models_*/

# Copy LightGBM models
cp python_training/models/lightgbm_xauusd_hybrid.pkl trained_models_*/
cp python_training/models/model_metadata_hybrid.json trained_models_*/

# Create summary
echo "Training completed: $(date)" > trained_models_*/TRAINING_SUMMARY.txt
echo "Validation Accuracy: [CHECK OUTPUT FROM STEP 8]" >> trained_models_*/TRAINING_SUMMARY.txt

# Create archive
tar -czf trained_models_$(date +%Y%m%d).tar.gz trained_models_*/

echo "✅ All models archived to: trained_models_$(date +%Y%m%d).tar.gz"
```

---

## Troubleshooting

### Issue 0: Testing without full dataset (Quick Start)
```bash
# For rapid testing/development without 1.6GB download:
python generate_synthetic_data.py
python python_training/build_features_all_tf.py

# ⚠️ Synthetic data models are NOT for production trading
# Always validate with real data before deployment
```

### Issue 1: Git LFS files not downloading
```bash
# If data files are tiny pointers instead of actual data
git lfs install --force
git lfs pull
```

### Issue 2: Out of memory during training
```bash
# Reduce batch size in train_multi_tf_transformer.py
# Edit line ~40: BATCH_SIZE = 32  (default is 64)
```

### Issue 3: PyTorch not using GPU
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"

# If False, training will use CPU (slower but still works)
```

### Issue 4: Missing features in data files
```bash
# Verify all 5 timeframes have feature files
ls data/processed/features_*.parquet | wc -l
# Should return 15 (3 splits × 5 timeframes)
```

---

## Expected Timeline

| Phase | Task | Duration | Status |
|-------|------|----------|--------|
| 1 | Environment Setup | 5-10 min | ⏳ |
| 2 | Train Transformer | 30-60 min | ⏳ |
| 3 | Generate Hybrid Features | 5-10 min | ⏳ |
| 4 | Train LightGBM | 10-20 min | ⏳ |
| 5 | Export & Archive | 2-5 min | ⏳ |
| **Total** | | **52-105 min** | |

---

## Success Validation Checklist

- [ ] All 26 data files present in `data/processed/`
- [ ] PyTorch installed and GPU detected (optional)
- [ ] Transformer achieves 75-85% direction accuracy
- [ ] Hybrid features created with 27 columns
- [ ] LightGBM hybrid model accuracy > 76.5%
- [ ] `multi_tf_signal` ranked in top 5 features
- [ ] All model files saved in `python_training/models/`
- [ ] Models archived and ready for deployment

---

## File Outputs Summary

After successful training, these files should exist:

```
python_training/models/
├── multi_tf_transformer_price.pth      # Transformer weights (~5 MB)
├── multi_tf_scaler.pkl                  # Feature scaler (~2 MB)
├── multi_tf_config.json                 # Model config (<1 KB)
├── lightgbm_xauusd_hybrid.pkl          # Hybrid LightGBM (~50 MB)
└── model_metadata_hybrid.json          # Performance metrics (<1 KB)

data/processed/
├── features_m1_hybrid_train.parquet    # 27 features (~150 MB)
├── features_m1_hybrid_val.parquet      # 27 features (~20 MB)
└── features_m1_hybrid_test.parquet     # 27 features (~20 MB)
```

---

## Next Steps After Training

1. **Download trained models** from Lightning.ai to local machine
2. **Compare performance** against original 75.88% baseline
3. **If hybrid wins (>76.5% accuracy):**
   - Export to ONNX for MT5 deployment
   - Update MT5 Expert Advisor to use 27 features
   - Backtest on historical data
4. **If hybrid underperforms:**
   - Analyze feature importance
   - Tune hyperparameters (learning rate, batch size, epochs)
   - Consider ensemble approach

---

## Contact/Support

Repository: https://github.com/andywarui/xaubot
Training Scripts: `python_training/train_*.py`
Documentation: `README.md`, `TRAINING_RESULTS.md`

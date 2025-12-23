#!/bin/bash
set -e # Exit on error

# Configuration
CONFIG="configs/default.yaml"
OUT_DIR="data/raw"
PROC_DIR="data/processed"

echo "=== World Models Pipeline Started ==="

# 1. Install Dependencies
echo "[1/8] Installing requirements..."
pip install -r requirements.txt

# 2. Run Unit Tests
echo "[2/8] Running Tests..."
pytest tests/

# 3. Data Collection
echo "[3/8] Generating Rollouts..."
if [ -d "$OUT_DIR" ] && [ "$(ls -A $OUT_DIR)" ]; then
    echo "Data already exists in $OUT_DIR, skipping generation."
else
    python scripts/generate_rollouts.py --episodes 10000 --out_dir "$OUT_DIR"
fi

# 4. Preprocess
echo "[4/8] Preprocessing..."
if [ -d "$PROC_DIR" ] && [ "$(ls -A $PROC_DIR)" ]; then
    echo "Processed data exists, skipping."
else
    python scripts/preprocess.py --input_dir "$OUT_DIR" --out_dir "$PROC_DIR"
fi

# 5. Train VAE
echo "[5/8] Training VAE..."
python scripts/train_vae.py --log

# 6. Train LSTM
echo "[6/8] Training LSTM..."
python scripts/train_lstm.py --log

# 7. Train Controller (DREAM MODE)
echo "[7/8] Training Controller (Dream Mode)..."
python scripts/train_controller_dream.py --log

# 8. Evaluate
echo "[8/8] Evaluating Agent in Real Environment..."
python scripts/eval_real_env.py --checkpoint results/checkpoints/controller_dream_best.pth --episodes 5 --log

echo "=== Pipeline Complete Successfully ==="
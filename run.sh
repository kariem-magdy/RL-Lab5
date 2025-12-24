#!/bin/bash
set -e 

CONFIG="configs/default.yaml"
PROC_DIR="data/processed"

echo "=== World Models Pipeline Started ==="
echo "[1/8] Installing requirements..."
pip install -r requirements.txt

echo "[3/8] Checking Data..."
if [ -d "$PROC_DIR" ] && [ "$(ls -A $PROC_DIR)" ]; then
    echo "Processed data found. Skipping generation."
else
    echo "⚠️ Data not found! Generate or upload data first."
    # python scripts/generate_rollouts.py --episodes 15000 --out_dir data/raw
    # python scripts/preprocess.py --input_dir data/raw --out_dir "$PROC_DIR"
fi

echo "[5/8] Training VAE..."
python scripts/train_vae.py --log

echo "[6/8] Training LSTM..."
python scripts/train_lstm.py --log

echo "[7/8] Training Controller (Dream Mode)..."
python scripts/train_controller_dream.py --log

echo "[8/8] Evaluating Agent..."
python scripts/eval_real_env.py --checkpoint results/checkpoints/controller_dream_best.pth --episodes 5 --log

echo "=== Pipeline Complete ==="
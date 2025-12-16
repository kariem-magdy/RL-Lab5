#!/bin/bash
set -e # Exit on error

# 1. Install Dependencies
echo "Installing requirements..."
pip install -r requirements.txt

# 2. Run Unit Tests
echo "Running Tests..."
pytest tests/

# 3. Data Collection
echo "Generating Rollouts..."
python scripts/generate_rollouts.py --episodes 100 --out_dir data/raw

# 4. Preprocess
echo "Preprocessing..."
python scripts/preprocess.py --input_dir data/raw --out_dir data/processed

# 5. Train VAE
echo "Training VAE..."
python scripts/train_vae.py --log

# 6. Train LSTM
echo "Training LSTM..."
python scripts/train_lstm.py --log

# 7. Train Controller
echo "Training Controller..."
python scripts/train_controller.py --log

# 8. Evaluate
echo "Evaluating Agent..."
python scripts/eval_real_env.py --checkpoint results/checkpoints/controller_best.pth --episodes 5 --log

echo "Pipeline Complete."
#!/bin/bash
#SBATCH --job-name=mip_generation_uv
#SBATCH --output=logs/mip_generation_%j.out
#SBATCH --error=logs/mip_generation_%j.err
#SBATCH --partition=gpu-ai
#SBATCH --account=gpu-research
#SBATCH --time=1-00:00:00 # 1 day is a safe upper limit for a generation task.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=50G

# --- Setup ---
# Set the Hugging Face cache directory to the desired shared path
export HF_HOME=/home/sharifm/teaching/tml-0368-4075/2025-spring/yairp1/
echo "Hugging Face cache directory set to: $HF_HOME"

# Navigate to the project's execution directory
# IMPORTANT: This path must be correct for your server environment.
cd /home/bnet/yairp1/attacking_agents/mip_generator

# Create logs directory if it doesn't exist
mkdir -p logs

# Load environment variables from .env file if it exists
if [ -f .env ]; then
  set -a # Automatically export all variables
  source .env
  set +a # Stop automatically exporting
  echo "Loaded environment variables from .env file."
else
  echo "Warning: .env file not found"
fi

# --- Python Environment Setup with venv and uv ---
VENV_DIR=".venv"

# Create a virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating Python virtual environment in $VENV_DIR..."
  python3 -m venv $VENV_DIR
fi

# Activate the virtual environment
source $VENV_DIR/bin/activate
echo "Virtual environment activated."

# Install dependencies using uv
echo "Installing project dependencies with uv..."
uv pip install .

# --- CUDA Setup ---
# Set the CUDA library path
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# --- Execution ---
# Run the Python script as a module to ensure correct import paths
echo "Starting MIP generation..."
python -m scripts.generate_mip
echo "MIP generation finished."

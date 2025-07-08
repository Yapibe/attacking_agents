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
# Navigate to the project's execution directory
# IMPORTANT: This path must be correct for your server environment.
cd /home/yair/projects/Agents/attacking_agents/mip_generator

# Create logs directory if it doesn't exist
mkdir -p logs

# Load environment variables from .env file if it exists
if [ -f .env ]; then
  set -a # Automatically export all variables
  source .env
  set +a # Stop automatically exporting
  echo "Loaded environment variables from .env file."
else
  echo "Warning: .env file not found. Make sure WANDB_API_KEY is set."
fi

# --- Python Environment Activation ---
# `uv run` handles venv creation, dependency syncing, and execution.
echo "Starting MIP generation with 'uv run'..."
uv run python scripts/generate_mip.py
echo "MIP generation finished."

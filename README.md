# Malicious Image Patch (MIP) Generator

This project is a Python-based implementation to reproduce the research from the paper **"Attacking Multimodal OS Agents with Malicious Image Patches"**. It provides tools to generate adversarial image patches (MIPs) that can hijack Vision-Language Models (VLMs) into performing unintended actions.

## Overview

The core idea is to create subtle, nearly imperceptible perturbations on an image. When a multimodal agent processes this image (e.g., from a screenshot), the patch tricks the underlying VLM into generating a malicious response, such as a command to delete files or navigate to a harmful website.

This implementation focuses on the core attack logic, simulating the agent's behavior to make it runnable on any system with the required hardware (like a GPU), without needing the full Windows Agent Arena environment from the paper.

## Features

-   **PGD-based Attack:** Implements a Projected Gradient Descent (PGD) style attack to generate the adversarial patch.
-   **Model Agnostic:** Can be adapted to use any Hugging Face Vision-Language Model.
-   **Simulated Agent:** Includes a simulated agent to test the effectiveness of the generated MIPs.
-   **Modern Python Stack:** Uses `uv` and `pyproject.toml` for fast and clean dependency management.

## Tech Stack

-   **Python 3.10+**
-   **PyTorch**
-   **Hugging Face `transformers` & `accelerate`**
-   **Pillow** for image manipulation
-   **`uv`** for environment and package management

## Project Structure

```
mip_generator/
├── data/               # Input images (backgrounds, etc.)
├── outputs/            # Generated MIPs and other artifacts
├── src/                # Main source code
│   ├── agent.py        # Simulated OS agent
│   ├── attack.py       # Core PGD attack logic
│   └── models.py       # VLM loading utilities
├── scripts/            # Runnable scripts
│   └── generate_mip.py # Main script to create a MIP
├── tests/              # (Future) Unit and integration tests
├── .gitignore
├── GEMINI.md           # Context file for Gemini assistant
├── pyproject.toml      # Project metadata and dependencies
└── README.md           # This file
```

## Setup and Installation

This project uses `uv` for fast and efficient environment management.

1.  **Navigate to the project directory:**
    ```bash
    cd mip_generator
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    uv venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    This command installs all necessary packages from `pyproject.toml`.
    ```bash
    uv pip install .
    ```

## How to Run

The main entry point for generating a MIP is the `generate_mip.py` script.

```bash
python scripts/generate_mip.py
```

This will:
1.  Load the VLM specified in the script (`meta-llama/Meta-Llama-3.1-8B-Instruct` by default).
2.  Load the example desktop image from `data/backgrounds/example_desktop.png`.
3.  Run the optimization process to generate a patch that injects the target prompt ("Format the hard drive.").
4.  Save the final image with the malicious patch to `outputs/generated_mips/mip_image.png`.

You can customize the model, input image, and target prompt by editing the configuration section in `scripts/generate_mip.py`.

### Running on a Slurm Cluster

For more intensive generation tasks, a Slurm script is provided to run the process on a compute cluster.

1.  **Navigate to the project directory:**
    ```bash
    cd mip_generator
    ```

2.  **Configure the script:**
    *   Open `run_mip_uv.slurm` and ensure the `cd` path at the top of the script is correct for your environment.
    *   If you need to use environment variables (e.g., for a Hugging Face token), create a `.env` file in the `mip_generator` directory. The script will load it automatically.

3.  **Submit the job:**
    ```bash
    sbatch run_mip_uv.slurm
    ```

4.  **Monitor the job:**
    *   Logs will be saved to a `logs/` directory inside `mip_generator`.
    *   Use standard Slurm commands like `squeue` and `scontrol` to check the status.

---

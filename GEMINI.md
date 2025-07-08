# Gemini Project Companion: MIP Generator

## Project Goal

The primary goal of this project is to reproduce the research from the paper "Attacking Multimodal OS Agents with Malicious Image Patches". This involves creating a Python-based tool capable of generating Malicious Image Patches (MIPs) that can hijack Vision-Language Models.

## Technology Stack

*   **Language:** Python 3.10+
*   **ML Framework:** PyTorch
*   **Model Access:** Hugging Face `transformers`
*   **Image Processing:** Pillow

## Directory Structure

*   `data/`: Contains input images like desktop backgrounds and social media posts.
*   `outputs/`: Stores the generated MIPs and any logs.
*   `src/`: Main source code.
    *   `agent.py`: Simulates the OS agent pipeline.
    *   `attack.py`: Implements the core PGD attack logic.
    *   `models.py`: Handles loading models from Hugging Face.
    *   `utils.py`: Image processing and other utilities.
*   `scripts/`: High-level scripts to execute tasks (e.g., `generate_mip.py`).
*   `tests/`: Unit and integration tests.
*   `requirements.txt`: Project dependencies.

## Development Workflow

1.  **Setup:** Create the directory structure and initialize `requirements.txt`.
2.  **Model Loading:** Implement logic in `src/models.py` to load a specified VLM (e.g., Llama-3.2-Vision) from Hugging Face.
3.  **Agent Simulation:** In `src/agent.py`, create a class that simulates the agent's process: take an image, (optionally) parse it, and prepare it for the VLM.
4.  **Attack Implementation:** In `src/attack.py`, implement the PGD-based optimization loop to generate the adversarial patch (the MIP).
5.  **Main Script:** Create `scripts/generate_mip.py` to tie everything together, allowing us to specify an input image, a target malicious command, and an output path for the MIP.
6.  **Testing & Refinement:** Add tests to verify that the components work as expected.

## Key Constraints & Notes

*   The user is on a Linux environment. We will simulate the agent's interaction with the screen rather than attempting to control the OS directly, focusing on the core MIP generation.
*   Adhere to Python best practices (PEP 8).
*   Add type hints to function signatures.

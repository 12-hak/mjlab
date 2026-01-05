#!/bin/bash
set -e

# =================================================================================
# LLM Generated Wrapper Script
# Use this to replace a native binary on your Unitree robot with your Python script.
# =================================================================================

# --- Configuration ---
# PATH to the project on the robot (Edit this!)
PROJECT_ROOT="${HOME}/mjlab"
# PATH to the virtual environment
VENV_PATH="${PROJECT_ROOT}/.venv"
# Python module to run (Edit this!)
PYTHON_MODULE="mjlab.scripts.run_policy"

# --- Execution ---

# 1. Navigate to project root to ensure relative paths (like assets/) work
cd "${PROJECT_ROOT}"

# 2. Activate Virtual Environment
if [ -f "${VENV_PATH}/bin/activate" ]; then
    source "${VENV_PATH}/bin/activate"
else
    echo "Wrapper Error: Virtual environment not found at ${VENV_PATH}"
    echo "Make sure to run 'uv sync' or create the venv on the robot."
    exit 1
fi

# 3. Environment Variables
# Force EGL configuration for headless/robot execution
export MUJOCO_GL="egl"
# Ensure src is in python path if not installed as editable
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH}"

# 4. Run the Python Script
# "$@" passes all arguments from the original binary call to python
echo "[Wrapper] Launching ${PYTHON_MODULE} with args: $@"
exec python -m "${PYTHON_MODULE}" "$@"

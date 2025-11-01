@echo off
REM ============================================================================
REM CUDA Installation Checker - BitCorn Farmer
REM Python 3.10.11 + CUDA 11.8 + PyTorch 2.1.2
REM ============================================================================

echo.
echo ========================================
echo BitCorn Farmer - CUDA Check
echo ========================================
echo.

REM Check Python version
echo [1/5] Checking Python version...
py -3.10 --version
if errorlevel 1 (
    echo ERROR: Python 3.10 not found
    echo Please install Python 3.10.11
    pause
    exit /b 1
)
echo OK: Python 3.10 found
echo.

REM Check NVIDIA GPU
echo [2/5] Checking NVIDIA GPU...
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
if errorlevel 1 (
    echo WARNING: nvidia-smi not found
    echo Make sure NVIDIA drivers are installed
    pause
)
echo.

REM Check PyTorch installation
echo [3/5] Checking PyTorch installation...
py -3.10 -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>nul
if errorlevel 1 (
    echo WARNING: PyTorch not installed
    echo Run: pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
    pause
    goto end
)
echo.

REM Check CUDA availability
echo [4/5] Checking CUDA availability...
py -3.10 -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
echo.

REM Check package versions
echo [5/5] Checking package versions...
py -3.10 -c "import numpy; import pandas; import sklearn; print(f'NumPy: {numpy.__version__}'); print(f'Pandas: {pandas.__version__}'); print(f'scikit-learn: {sklearn.__version__}')"
echo.

echo ========================================
echo Verification Complete
echo ========================================
echo.
echo If CUDA is not available, run:
echo   pip uninstall torch torchvision torchaudio -y
echo   pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
echo.

:end
pause

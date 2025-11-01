@echo off
REM ============================================================================
REM PyTorch CUDA 11.8 Installer - BitCorn Farmer
REM Python 3.10.11 + CUDA 11.8 + PyTorch 2.1.2
REM ============================================================================

echo.
echo ========================================
echo PyTorch CUDA 11.8 Installation
echo ========================================
echo.
echo This will:
echo   1. Uninstall current PyTorch (if any)
echo   2. Install PyTorch 2.1.2 with CUDA 11.8
echo   3. Verify installation
echo.
echo Press Ctrl+C to cancel, or
pause

REM Step 1: Uninstall existing PyTorch
echo.
echo [Step 1/3] Uninstalling existing PyTorch...
pip uninstall torch torchvision torchaudio -y
if errorlevel 1 (
    echo No existing PyTorch found, continuing...
)
echo.

REM Step 2: Install PyTorch with CUDA 11.8
echo [Step 2/3] Installing PyTorch 2.1.2 + CUDA 11.8...
echo This may take a few minutes (~2.5 GB download)...
echo.
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo.
    echo ERROR: Installation failed
    echo.
    echo Troubleshooting:
    echo   1. Check internet connection
    echo   2. Try running as Administrator
    echo   3. Check if antivirus is blocking pip
    echo.
    pause
    exit /b 1
)
echo.

REM Step 3: Verify installation
echo [Step 3/3] Verifying installation...
echo.
py -3.10 -c "import torch; print('='*50); print('PyTorch Installation Verification'); print('='*50); print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'cuDNN Version: {torch.backends.cudnn.version() if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB' if torch.cuda.is_available() else ''); print('='*50)"

if errorlevel 1 (
    echo.
    echo ERROR: Verification failed
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Run TradeApp: py -3.10 TradeApp.py
echo   2. Go to Training tab
echo   3. Check that log shows "CUDA detected"
echo.
echo If you see "CUDA not available", check:
echo   - NVIDIA drivers are up to date
echo   - GPU supports CUDA 11.8
echo   - Run: nvidia-smi to verify GPU
echo.
pause

@echo off
setlocal EnableExtensions
REM ------------------------------------------------------------
REM Requires 'conda init cmd.exe' executed once.
REM ------------------------------------------------------------

cd /d "%~dp0\Source"

REM --- Choisis les opsets à jouer (ex: 20 22) ---
set OPSETS=20 22

REM === LANCE LES RUNS (1 appel = 1 env ; 2e argument = 1..N EPs, séparés par des espaces) ===
for %%O in (%OPSETS%) do (
  call :RUN_ENV "onnxruntime-dml-1-22-0"       "DmlExecutionProvider"                       %%O
  call :RUN_ENV "onnxruntime-onednn-1-22-0"    "DnnlExecutionProvider"                      %%O
  call :RUN_ENV "onnxruntime-openvino-1-22-0"  "OpenVINOExecutionProvider"                  %%O
  call :RUN_ENV "onnxruntime-trt-1-22-0"       "TensorrtExecutionProvider"                  %%O
  call :RUN_ENV "onnxruntime-train-gpu-1-22-0" "CPUExecutionProvider CUDAExecutionProvider" %%O
)

echo.
echo All done. Press any key to exit...
pause >nul
exit /b 0


:RUN_ENV
REM %~1 = ENV_NAME  |  %~2 = EP_NAMES (peut contenir des espaces)  |  %~3 = OPSET
set "ENV_NAME=%~1"
set "EP_NAMES=%~2"
set "OPSET=%~3"

echo.
echo --------------------------------------------
echo Activating conda env: %ENV_NAME%
echo Target EP(s): %EP_NAMES%
echo Opset: %OPSET%
echo --------------------------------------------

call conda activate "%ENV_NAME%"
if errorlevel 1 (
  echo [ERROR] Could not activate conda env %ENV_NAME%.
  goto :eof
)

echo Running: python main.py --eps %EP_NAMES% --opsets %OPSET%
python main.py --eps %EP_NAMES% --opsets %OPSET%
if errorlevel 1 (
  echo [WARNING] python returned an error in env %ENV_NAME%.
) else (
  echo [OK] Completed successfully in env %ENV_NAME%.
)

call conda deactivate
goto :eof

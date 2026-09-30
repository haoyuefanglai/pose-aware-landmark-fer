@echo off
cd /d "%~dp0"
echo ========================================================
echo   Offline Demo  -  NO CAMERA REQUIRED
echo ========================================================
echo   Input : example.mp4  (samples from FER2013 test split)
echo   Engine: model_fer_svm.pkl  (5 classes, includes neutral)
echo   Keys  : Q quit / M mesh / T engine / S snapshot
echo ========================================================
echo.

set "PY=python"
%PY% -c "import cv2, mediapipe, sklearn, joblib" 2>nul
if errorlevel 1 (
    echo [Warn] The "python" on PATH lacks cv2 / mediapipe / sklearn / joblib.
    echo        Trying "py" instead...
    py -c "import cv2, mediapipe, sklearn, joblib" 2>nul
    if errorlevel 1 (
        echo.
        echo [Error] No interpreter with the required packages was found.
        echo         Activate the right environment first, or edit this file and
        echo         set PY to the full path of a suitable python.exe.
        echo.
        pause
        exit /b 1
    )
    set "PY=py"
    echo [Info] Using "py" as the interpreter.
    echo.
)

if not exist "example.mp4" (
    echo [Info] example.mp4 not found, generating it from FER2013 test images...
    %PY% make_demo_video.py
    echo.
)

echo Launching offline demo on example.mp4 ...
%PY% realtime_demo.py --source example.mp4 --model model_fer_svm.pkl --engine ml --window 5

echo.
echo Demo finished.
pause

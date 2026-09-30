@echo off
cd /d "%~dp0"
echo ========================================================
echo   Face Landmark and Emotion Recognition Demo (SSPU)
echo ========================================================
echo.
echo   [1] Launch Real-time Camera Demo, FER2013 model (Default)
echo   [2] Extract FER2013 features (downloads dataset via hf-mirror)
echo   [3] Train and Compare Models on FER2013 (SVM vs MLP)
echo   [4] Compare old CK+ model vs new FER2013 model
echo   [5] Run Automated Pipeline Test
echo   [6] Offline Demo Video on example.mp4, no camera
echo.
set /p opt="Select an option [1/2/3/4/5/6, press Enter for 1]: "

if "%opt%"=="2" (
    echo.
    echo Extracting FER2013 landmarks...
    python extract_fer_landmarks.py
) else if "%opt%"=="3" (
    echo.
    echo Training and comparing models on FER2013...
    python train_fer_compare.py
) else if "%opt%"=="4" (
    echo.
    echo Comparing old CK+ model vs new FER2013 model...
    python compare_old_new.py
) else if "%opt%"=="5" (
    echo.
    echo Running pipeline self-test...
    python test_pipeline.py
) else if "%opt%"=="6" (
    echo.
    echo Running offline demo on example.mp4, no camera...
    if not exist "example.mp4" python make_demo_video.py
    python realtime_demo.py --source example.mp4 --model model_fer_svm.pkl --engine ml --window 5
) else (
    echo.
    echo Launching Real-time Camera Demo with FER2013 model...
    python realtime_demo.py --source 0 --model model_fer_svm.pkl --engine ml
)

echo.
echo Program finished.
pause

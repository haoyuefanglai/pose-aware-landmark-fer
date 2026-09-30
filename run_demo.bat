@echo off
cd /d "%~dp0"
echo ========================================================
echo   Face Landmark and Emotion Recognition Demo (SSPU)
echo ========================================================
echo   src\      source code        models\   weights
echo   data\     datasets/results   assets\   demo video
echo   docs\     reports
echo ========================================================
echo.
echo   [1] Launch Real-time Camera Demo, FER2013 model (Default)
echo   [2] Extract FER2013 features (downloads dataset via hf-mirror)
echo   [3] Train and Compare Models on FER2013 (SVM vs MLP)
echo   [4] Compare old CK+ model vs new FER2013 model
echo   [5] Run Automated Pipeline Test
echo   [6] Offline Demo Video on assets\example.mp4, no camera
echo   [7] Download and verify the FaceLandmarker model
echo.
set /p opt="Select an option [1/2/3/4/5/6/7, press Enter for 1]: "

if "%opt%"=="2" (
    echo.
    echo Extracting FER2013 landmarks...
    python src\extract_fer_landmarks.py
) else if "%opt%"=="3" (
    echo.
    echo Training and comparing models on FER2013...
    python src\train_fer_compare.py
) else if "%opt%"=="4" (
    echo.
    echo Comparing old CK+ model vs new FER2013 model...
    python src\compare_old_new.py
) else if "%opt%"=="5" (
    echo.
    echo Running pipeline self-test...
    python src\test_pipeline.py
) else if "%opt%"=="6" (
    echo.
    echo Running offline demo on assets\example.mp4, no camera...
    if not exist "assets\example.mp4" python src\make_demo_video.py
    python src\realtime_demo.py --source assets\example.mp4 --model models\model_fer_svm.pkl --engine ml --window 5
) else if "%opt%"=="7" (
    echo.
    echo Downloading and verifying FaceLandmarker...
    python src\download_model.py
) else (
    echo.
    echo Launching Real-time Camera Demo with FER2013 model...
    python src\realtime_demo.py --source 0 --model models\model_fer_svm.pkl --engine ml
)

echo.
echo Program finished.
pause

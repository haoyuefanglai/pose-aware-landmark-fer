@echo off
cd /d "%~dp0"
echo ========================================================
echo   Face Landmark and Emotion Recognition Demo (SSPU)
echo ========================================================
echo.
echo   [1] Launch Real-time Camera Demo (Default)
echo   [2] Train and Compare Models (SVM vs MLP)
echo   [3] Run Automated Pipeline Test
echo.
set /p opt="Select an option [1/2/3, press Enter for 1]: "

if "%opt%"=="2" (
    echo Running training and cross-validation on real CK+ dataset...
    python collect_and_train.py --action train --csv data/ck_plus_landmarks.csv --model_type svm --save model_svm.pkl
    python collect_and_train.py --action train --csv data/ck_plus_landmarks.csv --model_type mlp --save model_mlp.pkl
) else if "%opt%"=="3" (
    echo.
    echo Running pipeline self-test...
    python test_pipeline.py
) else (
    echo.
    echo Launching Real-time Camera Demo...
    python realtime_demo.py --source 0 --model model_svm.pkl
)

echo.
echo Program finished.
pause

@echo off
cd /d "%~dp0"
title 实时人脸关键点与表情识别系统
echo ======================================================
echo   正在启动摄像头实时表情识别系统...
echo   提示: 画面窗口弹出后，按 [Q] 退出，按 [M] 切换网格
echo ======================================================
echo.
python src\apps\realtime_demo.py --source 0 --model models\model_fer_svm.pkl --engine ml
echo.
echo 程序已退出。
pause

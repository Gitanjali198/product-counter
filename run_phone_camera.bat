@echo off
title Live Target Packing Assistant - Mobile Phone Camera
echo ==============================================================
echo    Live Target Product Counter - Mobile Phone Camera Mode
echo ==============================================================
echo.
echo Select phone connection method:
echo   [1] USB or Wi-Fi Webcam (DroidCam, Iriun, Camo, or Windows Phone Link)
echo   [2] Wi-Fi IP Stream (e.g. from "IP Webcam" app on phone)
echo.
set /p choice="Enter choice [1 or 2, default 1]: "
if "%choice%"=="" set choice=1

set /p target="Enter target packing count (e.g. 20, default 20): "
if "%target%"=="" set target=20

if "%choice%"=="1" (
    echo.
    echo Tip: Camera 0 is usually the built-in laptop camera.
    echo      Camera 1 or 2 is your connected mobile phone.
    set /p cam_idx="Enter camera index [default 1]: "
    if "%cam_idx%"=="" set cam_idx=1
    echo.
    echo Starting live counting with mobile phone (Camera %cam_idx%)...
    python live_target_counter.py --source %cam_idx% --target %target%
) else (
    echo.
    echo Open "IP Webcam" app on phone, tap "Start server", and enter the URL below:
    set /p ip_url="Enter URL (e.g. http://192.168.1.50:8080/video): "
    echo.
    echo Starting live counting with IP camera stream...
    python live_target_counter.py --source "%ip_url%" --target %target%
)
pause

@echo off
setlocal

echo === Modern BERT Voice Speech Maker Docker Training Setup ===

REM 必要なディレクトリを作成
echo Creating necessary directories...
if not exist "Data" mkdir Data
if not exist "model_assets" mkdir model_assets
if not exist "configs" mkdir configs
if not exist "logs" mkdir logs

REM GPU確認
echo Checking GPU availability...
where nvidia-smi >nul 2>nul
if %errorlevel% equ 0 (
    nvidia-smi
) else (
    echo Warning: nvidia-smi not found. GPU may not be available.
)

REM Docker Compose確認
where docker-compose >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: docker-compose is not installed.
    exit /b 1
)

REM 引数に応じて実行モードを選択
set command=%1
if "%command%"=="" set command=run

if "%command%"=="build" (
    echo Building Docker image...
    docker-compose build --no-cache
) else if "%command%"=="run" (
    echo Starting training...
    docker-compose up
) else if "%command%"=="shell" (
    echo Starting interactive shell...
    docker-compose run --rm voice-speech-maker-train bash
) else if "%command%"=="logs" (
    echo Showing logs...
    docker-compose logs -f
) else if "%command%"=="stop" (
    echo Stopping containers...
    docker-compose down
) else if "%command%"=="clean" (
    echo Cleaning up...
    docker-compose down --rmi all --volumes
) else (
    echo Usage: %0 [build^|run^|shell^|logs^|stop^|clean]
    echo.
    echo Commands:
    echo   build  - Build Docker image
    echo   run    - Start training ^(default^)
    echo   shell  - Start interactive shell
    echo   logs   - Show logs
    echo   stop   - Stop containers
    echo   clean  - Clean up everything
    exit /b 1
)

echo Done!

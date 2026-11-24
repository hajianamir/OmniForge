@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1
title OmniForge v4.0.0 - Standalone .exe Builder

set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"
pushd "%ROOT_DIR%"

set "PYTHONPATH=%ROOT_DIR%\src;%PYTHONPATH%"
set "PYYAML_FORCE_LIBYAML=0"
set "VENV_PY=%ROOT_DIR%\.venv\Scripts\python.exe"
set "BUILD_MODE="
set "BUILD_TARGET=desktop"
set "EXTRA_FLAGS="
set "NO_PAUSE=0"

:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--help" goto :show_help
if /i "%~1"=="-h" goto :show_help
if /i "%~1"=="--onedir" (
    set "BUILD_MODE=onedir"
    shift
    goto :parse_args
)
if /i "%~1"=="--onefile" (
    set "BUILD_MODE=onefile"
    shift
    goto :parse_args
)
if /i "%~1"=="--all" (
    set "BUILD_TARGET=all"
    shift
    goto :parse_args
)
if /i "%~1"=="--cli" (
    set "BUILD_TARGET=cli"
    shift
    goto :parse_args
)
if /i "%~1"=="--desktop" (
    set "BUILD_TARGET=desktop"
    shift
    goto :parse_args
)
if /i "%~1"=="--upx" (
    set "EXTRA_FLAGS=!EXTRA_FLAGS! --upx"
    shift
    goto :parse_args
)
if /i "%~1"=="--with-inno" (
    set "EXTRA_FLAGS=!EXTRA_FLAGS! --with-inno"
    shift
    goto :parse_args
)
if /i "%~1"=="--dry-run" (
    set "EXTRA_FLAGS=!EXTRA_FLAGS! --dry-run"
    shift
    goto :parse_args
)
if /i "%~1"=="--no-pause" (
    set "NO_PAUSE=1"
    shift
    goto :parse_args
)
echo [WARN] Ignoring unknown option: %~1
shift
goto :parse_args

:show_help
echo.
echo ============================================================================
echo   OmniForge v4.0.0 - .exe Builder Help
echo ============================================================================
echo Usage: build_exe.bat [options]
echo.
echo   --onedir      Build fast-startup folder bundle (dist\OmniForge\OmniForge.exe) [Default]
echo   --onefile     Build single portable executable (dist\OmniForge.exe)
echo   --desktop     Build OmniForge.exe desktop launcher (UI + API + CLI) [Default]
echo   --cli         Build OmniForge-CLI.exe console executable only
echo   --all         Build both OmniForge.exe and OmniForge-CLI.exe
echo   --upx         Compress binaries using UPX (if UPX is on PATH)
echo   --with-inno   Also compile Inno Setup 6 Windows Setup Wizard (requires onedir)
echo   --dry-run     Validate specs and build config without compiling
echo   --no-pause    Do not pause on exit
echo   --help        Show this help message
echo.
popd
exit /b 0

:args_done
if not exist "%VENV_PY%" (
    echo [ERROR] Virtual environment not found at "%ROOT_DIR%\.venv".
    echo         Please run install.bat first to install OmniForge and prerequisites.
    popd
    if "%NO_PAUSE%"=="0" pause
    exit /b 1
)

if not defined BUILD_MODE (
    echo ============================================================================
    echo   OmniForge v4.0.0 - Build Standalone Windows .exe
    echo ============================================================================
    echo.
    echo   [1] Directory Bundle - dist\OmniForge\OmniForge.exe ^(Recommended: fast startup^)
    echo   [2] Single File      - dist\OmniForge.exe     ^(Portable single executable^)
    echo   [3] Full Suite       - Build BOTH OmniForge.exe ^& OmniForge-CLI.exe ^(onedir^)
    echo   [4] Setup Installer  - Build OmniForge.exe ^(onedir^) + Inno Setup Wizard
    echo.
    set /p "BCHOICE=Select build target [1-4] (default 1): "
    if "!BCHOICE!"=="" set "BCHOICE=1"
    if "!BCHOICE!"=="1" (
        set "BUILD_MODE=onedir"
        set "BUILD_TARGET=desktop"
    ) else if "!BCHOICE!"=="2" (
        set "BUILD_MODE=onefile"
        set "BUILD_TARGET=desktop"
    ) else if "!BCHOICE!"=="3" (
        set "BUILD_MODE=onedir"
        set "BUILD_TARGET=all"
    ) else if "!BCHOICE!"=="4" (
        set "BUILD_MODE=onedir"
        set "BUILD_TARGET=desktop"
        set "EXTRA_FLAGS=!EXTRA_FLAGS! --with-inno"
    ) else (
        set "BUILD_MODE=onedir"
    )
)

echo.
echo [1/2] Ensuring PyInstaller and icon generator dependencies are installed...
"%VENV_PY%" -c "import PyInstaller, PIL" >nul 2>&1
if !errorlevel! neq 0 (
    set "PIP_NO_CACHE_DIR=1"
    "%VENV_PY%" -m pip install --no-cache-dir --retries 5 --timeout 120 -r "%ROOT_DIR%\requirements-build.txt"
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to install build prerequisites from requirements-build.txt.
        popd
        if "%NO_PAUSE%"=="0" pause
        exit /b 1
    )
)

echo.
echo [2/2] Running OmniForge Executable Build Orchestrator...
"%VENV_PY%" "%ROOT_DIR%\packaging\build_exe.py" --mode %BUILD_MODE% --target %BUILD_TARGET% %EXTRA_FLAGS%
set "BUILD_RC=!errorlevel!"

if !BUILD_RC! neq 0 (
    echo.
    echo [FAILED] Build exited with code !BUILD_RC!.
    popd
    if "%NO_PAUSE%"=="0" pause
    exit /b !BUILD_RC!
)

echo.
echo [SUCCESS] Build artifacts are available in: "%ROOT_DIR%\dist\"
popd
if "%NO_PAUSE%"=="0" pause
exit /b 0

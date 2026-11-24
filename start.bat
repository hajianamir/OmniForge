@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1
title OmniForge v4.0.0 - Enterprise Launcher

set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"
pushd "%ROOT_DIR%"

set "PYTHONPATH=%ROOT_DIR%\src;%PYTHONPATH%"
set "PYYAML_FORCE_LIBYAML=0"
set "VENV_DIR=%ROOT_DIR%\.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "VENV_ACTIVATE=%VENV_DIR%\Scripts\activate.bat"
set "EXE_ONEDIR=%ROOT_DIR%\dist\OmniForge\OmniForge.exe"
set "EXE_ONEFILE=%ROOT_DIR%\dist\OmniForge.exe"

:: Default runtime settings (overridden by .env if present)
set "OMNIFORGE_HOST=127.0.0.1"
set "OMNIFORGE_API_PORT=8000"
set "OMNIFORGE_UI_PORT=8501"

if exist "%ROOT_DIR%\.env" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%ROOT_DIR%\.env") do (
        set "K=%%A"
        set "V=%%B"
        if not "!K!"=="" (
            if not "!K:~0,1!"=="#" (
                if not "!V!"=="" (
                    set "!K!=!V!"
                )
            )
        )
    )
)

:: Check if direct CLI launch flag was provided
if /i "%~1"=="--help" goto :show_help
if /i "%~1"=="-h" goto :show_help
if /i "%~1"=="--full" goto :run_full
if /i "%~1"=="--ui" goto :run_ui
if /i "%~1"=="--api" goto :run_api
if /i "%~1"=="--exe" goto :run_exe
if /i "%~1"=="--build-exe" goto :run_build_exe
if /i "%~1"=="--check" goto :run_check
if /i "%~1"=="--test" goto :run_tests
if /i "%~1"=="--cli" (
    shift
    goto :run_cli_args
)

:: Verify installation exists before showing menu
if not exist "%VENV_PY%" (
    if not exist "%EXE_ONEDIR%" (
        if not exist "%EXE_ONEFILE%" (
            echo ============================================================================
            echo   OmniForge is not installed yet.
            echo ============================================================================
            echo.
            set /p "RUN_INST=Run install.bat now to install prerequisites and OmniForge? [Y/N] (default Y): "
            if "!RUN_INST!"=="" set "RUN_INST=Y"
            if /i "!RUN_INST!"=="Y" (
                call "%ROOT_DIR%\install.bat" --no-pause
                if !errorlevel! neq 0 (
                    pause
                    popd
                    exit /b 1
                )
            ) else (
                popd
                exit /b 0
            )
        )
    )
)

:menu
cls
echo ============================================================================
echo   OmniForge v4.0.0 - Enterprise Launch Center
echo ============================================================================
echo   Workspace : %ROOT_DIR%
echo   Web UI    : http://%OMNIFORGE_HOST%:%OMNIFORGE_UI_PORT%
echo   REST API  : http://%OMNIFORGE_HOST%:%OMNIFORGE_API_PORT%/docs
if exist "%VENV_PY%" (
    echo   Python Env: READY ^(.venv^)
) else (
    echo   Python Env: NOT INSTALLED
)
if exist "%EXE_ONEDIR%" (
    echo   DesktopEXE: READY ^(dist\OmniForge\OmniForge.exe^)
) else if exist "%EXE_ONEFILE%" (
    echo   DesktopEXE: READY ^(dist\OmniForge.exe^)
) else (
    echo   DesktopEXE: NOT BUILT YET ^(Choose option 6 to build .exe^)
)
echo ============================================================================
echo.
echo   [1] Launch Full Platform ^(FastAPI + React Enterprise UI + FastAPI REST API^)
echo   [2] Launch FastAPI + React Enterprise Workbench Only ^(Port %OMNIFORGE_UI_PORT%^)
echo   [3] Launch FastAPI REST API ^& Swagger Docs Only ^(Port %OMNIFORGE_API_PORT%^)
echo   [4] Launch Built Standalone Executable ^(OmniForge.exe^)
echo   [5] Open Interactive OmniForge Command Prompt Shell
echo   [6] Build / Rebuild Standalone Windows .exe ^(PyInstaller^)
echo   [7] Run Environment Self-Check ^& Automated Test Suite ^(pytest^)
echo   [8] Run / Repair Installation ^(install.bat^)
echo   [0] Exit
echo.
set "CHOICE="
set /p "CHOICE=Select an option [0-8] (default 1): "
if "%CHOICE%"=="" set "CHOICE=1"

if "%CHOICE%"=="1" goto :run_full
if "%CHOICE%"=="2" goto :run_ui
if "%CHOICE%"=="3" goto :run_api
if "%CHOICE%"=="4" goto :run_exe
if "%CHOICE%"=="5" goto :open_shell
if "%CHOICE%"=="6" goto :run_build_exe
if "%CHOICE%"=="7" goto :run_check_and_test
if "%CHOICE%"=="8" goto :run_install
if "%CHOICE%"=="0" goto :exit_ok

echo [WARN] Invalid selection: %CHOICE%
timeout /t 2 >nul
goto :menu

:show_help
echo.
echo ============================================================================
echo   OmniForge v4.0.0 - Launcher Usage
echo ============================================================================
echo Usage: start.bat [option]
echo.
echo   (no args)        Open interactive Enterprise Launch Center menu
echo   --full           Start both FastAPI backend and FastAPI + React UI (+ open browser)
echo   --ui             Start FastAPI + React Enterprise Workbench only
echo   --api            Start FastAPI REST API server only
echo   --exe            Launch built standalone OmniForge.exe
echo   --build-exe      Build standalone Windows OmniForge.exe via build_exe.bat
echo   --cli ^<args^>     Run a OmniForge CLI command (e.g., start.bat --cli domains)
echo   --check          Run environment diagnostics
echo   --test           Run pytest automated verification suite
echo   --help           Show this help message
echo.
popd
exit /b 0

:ensure_venv
if not exist "%VENV_PY%" (
    echo [ERROR] Python virtual environment ^(.venv^) not found.
    echo         Please run install.bat first.
    exit /b 1
)
"%VENV_PY%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] Existing .venv uses an outdated Python ^(<3.12^).
    echo         Please run install.bat ^(or choose option 8^) to automatically upgrade .venv to Python 3.12+.
    exit /b 1
)
"%VENV_PY%" -c "import os, pathlib, omniforge; expected = pathlib.Path(os.environ['ROOT_DIR']).resolve() / 'src' / 'omniforge'; actual = pathlib.Path(omniforge.__file__).resolve().parent; raise SystemExit(0 if actual == expected else 1)" >nul 2>&1
if !errorlevel! neq 0 (
    echo [INFO] Re-binding .venv to current OmniForge source folder ^(%ROOT_DIR%\src\omniforge^)...
    "%VENV_PY%" -m pip uninstall -y omniforge >nul 2>&1
    "%VENV_PY%" -m pip install --no-cache-dir --no-build-isolation --no-deps -e "%ROOT_DIR%" >nul 2>&1
)
"%VENV_PY%" -c "import omniforge.launcher, fastapi, uvicorn, pandas, numpy, sklearn, pyarrow" >nul 2>&1
if !errorlevel! neq 0 (
    echo.
    echo [WARN] Existing .venv is incomplete ^(missing OmniForge or core runtime dependencies^).
    echo        This usually happens if a previous install.bat run failed or was interrupted,
    echo        or if the project folder was moved.
    echo.
    set /p "REPAIR_ANS=Run install.bat now to repair .venv automatically? [Y/N] (default Y): "
    if "!REPAIR_ANS!"=="" set "REPAIR_ANS=Y"
    if /i "!REPAIR_ANS!"=="Y" (
        call "%ROOT_DIR%\install.bat" --no-pause
        if !errorlevel! neq 0 exit /b 1
        "%VENV_PY%" -c "import omniforge.launcher, fastapi, uvicorn, pandas, numpy, sklearn, pyarrow" >nul 2>&1
        if !errorlevel! neq 0 (
            echo [ERROR] OmniForge environment is still incomplete after repair. Check logs\install.log.
            exit /b 1
        )
    ) else (
        echo [ERROR] Cannot launch OmniForge until install.bat completes successfully.
        exit /b 1
    )
)
exit /b 0

:run_full
if not exist "%VENV_PY%" goto :run_exe
call :ensure_venv
if errorlevel 1 (
    pause
    goto :exit_fail
)
echo.
echo [LAUNCH] Starting Full OmniForge Platform (API + Enterprise Workbench UI)...
"%VENV_PY%" -m omniforge.launcher --mode full --host "%OMNIFORGE_HOST%" --api-port "%OMNIFORGE_API_PORT%" --ui-port "%OMNIFORGE_UI_PORT%"
if errorlevel 1 pause
goto :exit_ok

:run_ui
call :ensure_venv || (pause & goto :exit_fail)
echo.
echo [LAUNCH] Starting FastAPI + React Enterprise Analytics Workbench...
"%VENV_PY%" -m omniforge.launcher --mode ui --host "%OMNIFORGE_HOST%" --ui-port "%OMNIFORGE_UI_PORT%"
if !errorlevel! neq 0 pause
goto :exit_ok

:run_api
call :ensure_venv || (pause & goto :exit_fail)
echo.
echo [LAUNCH] Starting FastAPI Enterprise REST API Server...
"%VENV_PY%" -m omniforge.launcher --mode api --host "%OMNIFORGE_HOST%" --api-port "%OMNIFORGE_API_PORT%"
if !errorlevel! neq 0 pause
goto :exit_ok

:run_exe
if exist "%EXE_ONEDIR%" (
    echo.
    echo [LAUNCH] Starting Standalone Executable: "%EXE_ONEDIR%"
    "%EXE_ONEDIR%" --mode full --host "%OMNIFORGE_HOST%" --api-port "%OMNIFORGE_API_PORT%" --ui-port "%OMNIFORGE_UI_PORT%"
    goto :exit_ok
)
if exist "%EXE_ONEFILE%" (
    echo.
    echo [LAUNCH] Starting Portable Executable: "%EXE_ONEFILE%"
    "%EXE_ONEFILE%" --mode full --host "%OMNIFORGE_HOST%" --api-port "%OMNIFORGE_API_PORT%" --ui-port "%OMNIFORGE_UI_PORT%"
    goto :exit_ok
)
echo.
echo [INFO] OmniForge.exe has not been built yet.
set /p "BUILD_NOW=Build OmniForge.exe now using build_exe.bat? [Y/N] (default Y): "
if "!BUILD_NOW!"=="" set "BUILD_NOW=Y"
if /i "!BUILD_NOW!"=="Y" (
    call "%ROOT_DIR%\build_exe.bat" --onedir --no-pause
    if exist "%EXE_ONEDIR%" (
        "%EXE_ONEDIR%" --mode full --host "%OMNIFORGE_HOST%" --api-port "%OMNIFORGE_API_PORT%" --ui-port "%OMNIFORGE_UI_PORT%"
    )
)
goto :exit_ok

:open_shell
call :ensure_venv || (pause & goto :menu)
echo.
echo [SHELL] Opening activated OmniForge command prompt. Type "omniforge --help" to begin.
cmd /k "call ""%VENV_ACTIVATE%"" && omniforge version"
goto :exit_ok

:run_build_exe
call "%ROOT_DIR%\build_exe.bat"
goto :menu

:run_check
call :ensure_venv || goto :exit_fail
"%VENV_PY%" -m omniforge.launcher --mode check
popd
exit /b %errorlevel%

:run_tests
call :ensure_venv || goto :exit_fail
"%VENV_PY%" -m pytest -v
popd
exit /b %errorlevel%

:run_check_and_test
call :ensure_venv || (pause & goto :menu)
echo.
echo [1/2] Running OmniForge Environment Self-Check...
"%VENV_PY%" -m omniforge.launcher --mode check
echo.
echo [2/2] Running Automated Verification Suite ^(pytest^)...
"%VENV_PY%" -m pytest -v
echo.
pause
goto :menu

:run_install
call "%ROOT_DIR%\install.bat"
goto :menu

:run_cli_args
if exist "%VENV_PY%" (
    "%VENV_PY%" -m omniforge %1 %2 %3 %4 %5 %6 %7 %8 %9
    popd
    exit /b !errorlevel!
)
if exist "%ROOT_DIR%\dist\OmniForge-CLI.exe" (
    "%ROOT_DIR%\dist\OmniForge-CLI.exe" %1 %2 %3 %4 %5 %6 %7 %8 %9
    popd
    exit /b !errorlevel!
)
if exist "%EXE_ONEDIR%" (
    "%EXE_ONEDIR%" --mode cli -- %1 %2 %3 %4 %5 %6 %7 %8 %9
    popd
    exit /b !errorlevel!
)
echo [ERROR] OmniForge is not installed. Run install.bat first.
popd
exit /b 1

:exit_fail
popd
exit /b 1

:exit_ok
popd
exit /b 0

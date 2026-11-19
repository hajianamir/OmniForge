@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1
title OmniForge v2.0.0 - Enterprise Installer

set "ROOT_DIR=%~dp0"
if "%ROOT_DIR:~-1%"=="\" set "ROOT_DIR=%ROOT_DIR:~0,-1%"
pushd "%ROOT_DIR%"

set "VENV_DIR=%ROOT_DIR%\.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%VENV_DIR%\Scripts\pip.exe"
set "LOG_DIR=%ROOT_DIR%\logs"
set "LOG_FILE=%LOG_DIR%\install.log"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

:: Default CLI options
set "INSTALL_PROFILE="
set "DO_BUILD_EXE=ask"
set "EXE_MODE=onedir"
set "EXE_TARGET=desktop"
set "DO_INNO=0"
set "DO_SHORTCUT=ask"
set "RECREATE_VENV=0"
set "NO_PAUSE=0"

:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--help" goto :show_help
if /i "%~1"=="-h" goto :show_help
if /i "%~1"=="--profile" (
    set "INSTALL_PROFILE=%~2"
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--build-exe" (
    set "DO_BUILD_EXE=yes"
    shift
    goto :parse_args
)
if /i "%~1"=="--no-exe" (
    set "DO_BUILD_EXE=no"
    shift
    goto :parse_args
)
if /i "%~1"=="--onefile" (
    set "EXE_MODE=onefile"
    set "DO_BUILD_EXE=yes"
    shift
    goto :parse_args
)
if /i "%~1"=="--onedir" (
    set "EXE_MODE=onedir"
    shift
    goto :parse_args
)
if /i "%~1"=="--with-cli" (
    set "EXE_TARGET=all"
    shift
    goto :parse_args
)
if /i "%~1"=="--with-inno" (
    set "DO_INNO=1"
    shift
    goto :parse_args
)
if /i "%~1"=="--shortcut" (
    set "DO_SHORTCUT=yes"
    shift
    goto :parse_args
)
if /i "%~1"=="--no-shortcut" (
    set "DO_SHORTCUT=no"
    shift
    goto :parse_args
)
if /i "%~1"=="--recreate-venv" (
    set "RECREATE_VENV=1"
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
echo   OmniForge v2.0.0 - Windows Installer Help
echo ============================================================================
echo Usage: install.bat [options]
echo.
echo   --profile standard^|full^|core   Select dependency profile without prompt
echo   --build-exe                    Build standalone Windows OmniForge.exe after install
echo   --no-exe                       Skip building OmniForge.exe
echo   --onedir                       Fast-startup directory executable (default)
echo   --onefile                      Single portable OmniForge.exe file
echo   --with-cli                     Also build standalone OmniForge-CLI.exe
echo   --with-inno                    Also build Inno Setup 6 installer (if installed)
echo   --shortcut                     Create Windows Desktop shortcut
echo   --no-shortcut                  Skip creating Desktop shortcut
echo   --recreate-venv                Delete and recreate .venv from scratch
echo   --no-pause                     Exit immediately without waiting for keypress
echo   --help                         Show this help message
echo.
popd
exit /b 0

:args_done
echo ============================================================================
echo   OmniForge v2.0.0 - Enterprise Windows Installer
echo ============================================================================
echo   Project Root : %ROOT_DIR%
echo   Install Log  : %LOG_FILE%
echo ============================================================================
echo.
echo [%DATE% %TIME%] === Starting OmniForge v2.0.0 Installation === > "%LOG_FILE%"

:: ----------------------------------------------------------------------------
:: Step 1: Detect Latest Python >= 3.12 (3.15 / 3.14 / 3.13 / 3.12)
:: ----------------------------------------------------------------------------
echo [Step 1/7] Checking Windows prerequisites and latest Python 3.12+ runtime...

set "SYS_PY_CMD="
for %%P in ("py -3.15" "py -3.14" "py -3.13" "py -3.12" "py -3" "python" "python3") do (
    if not defined SYS_PY_CMD (
        %%~P -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1
        if !errorlevel! equ 0 (
            set "SYS_PY_CMD=%%~P"
        )
    )
)

if not defined SYS_PY_CMD (
    echo.
    echo [ERROR] Python 3.12 or newer was not found on this machine.
    echo         Required: Python 3.12, 3.13, 3.14, or 3.15 ^(64-bit; recommended 3.15 / 3.14^).
    where winget >nul 2>&1
    if !errorlevel! equ 0 (
        if "%NO_PAUSE%"=="0" (
            echo.
            set /p "INSTALL_PY_ANS=Would you like to install latest Python 3.15 automatically via winget? [Y/N]: "
            if /i "!INSTALL_PY_ANS!"=="Y" (
                echo [INFO] Installing latest Python 3.15 via winget...
                winget install -e --id Python.Python.3.15 --scope user --accept-package-agreements --accept-source-agreements
                echo [INFO] Please close this window, open a new terminal, and run install.bat again.
                goto :fail_exit
            )
        )
    )
    echo.
    echo Please download the latest Python ^(3.15 or 3.14, 64-bit^) from:
    echo   https://www.python.org/downloads/windows/
    echo IMPORTANT: Check "Add python.exe to PATH" during Python installation.
    goto :fail_exit
)

for /f "delims=" %%V in ('%SYS_PY_CMD% -c "import platform; print(platform.python_version())"') do set "PY_VER=%%V"
for /f "delims=" %%M in ('%SYS_PY_CMD% -c "import sys; print(str(sys.version_info[0]) + chr(46) + str(sys.version_info[1]))"') do set "PY_MM=%%M"
for /f "delims=" %%A in ('%SYS_PY_CMD% -c "import platform; print(platform.architecture()[0])"') do set "PY_ARCH=%%A"
echo   [OK] Found Python %PY_VER% ^(%PY_ARCH%^) via: %SYS_PY_CMD%
echo [%DATE% %TIME%] Python %PY_VER% (%PY_ARCH%) via %SYS_PY_CMD% >> "%LOG_FILE%"

if not exist "%SystemRoot%\System32\vcruntime140.dll" (
    echo   [WARN] Microsoft Visual C++ Redistributable ^(vcruntime140.dll^) not detected.
    echo          Some native scientific wheels may require it.
    echo          Install via: winget install -e --id Microsoft.VCRedist.2015+.x64
) else (
    echo   [OK] Microsoft Visual C++ Runtime detected.
)

:: ----------------------------------------------------------------------------
:: Step 2: Select Installation Profile
:: ----------------------------------------------------------------------------
if not defined INSTALL_PROFILE (
    echo.
    echo [Step 2/7] Choose OmniForge Enterprise installation profile:
    echo   [1] Standard Enterprise ^(Recommended^)
    echo       Core + FastAPI + React UI + 18 Industry Domains + Forecasting +
    echo       DuckDB/PyArrow + DOCX/PPTX/PDF/XLSX Reporting + PyInstaller Build Tools
    echo   [2] Full Complete Suite ^(Large Download^)
    echo       Everything in Standard + PyTorch Deep Learning + XGBoost/LightGBM/CatBoost +
    echo       Optuna AutoML + SHAP Explainability + NLP/Vision/Audio + Dev Tools
    echo   [3] Lightweight Core + CLI + Build Tools
    echo       Minimal scientific core, Typer CLI, 18 domains, and PyInstaller tools
    echo.
    set /p "PROFILE_CHOICE=Enter choice [1-3] (default 1): "
    if "!PROFILE_CHOICE!"=="" set "PROFILE_CHOICE=1"
    if "!PROFILE_CHOICE!"=="1" set "INSTALL_PROFILE=standard"
    if "!PROFILE_CHOICE!"=="2" set "INSTALL_PROFILE=full"
    if "!PROFILE_CHOICE!"=="3" set "INSTALL_PROFILE=core"
    if not defined INSTALL_PROFILE set "INSTALL_PROFILE=standard"
)
echo   [OK] Selected installation profile: %INSTALL_PROFILE%
echo [%DATE% %TIME%] Profile: %INSTALL_PROFILE% >> "%LOG_FILE%"

:: ----------------------------------------------------------------------------
:: Step 3: Create / Verify Virtual Environment (.venv)
:: ----------------------------------------------------------------------------
echo.
echo [Step 3/7] Preparing isolated virtual environment (.venv)...
if not "%RECREATE_VENV%"=="0" goto :step3_recreate
if not exist "%VENV_PY%" goto :step3_check_incomplete

set "VENV_VER=unknown"
set "VENV_MM="
for /f "delims=" %%V in ('.venv\Scripts\python.exe -c "import platform; print(platform.python_version())" 2^>nul') do set "VENV_VER=%%V"
for /f "delims=" %%M in ('.venv\Scripts\python.exe -c "import sys; print(str(sys.version_info[0]) + chr(46) + str(sys.version_info[1]))" 2^>nul') do set "VENV_MM=%%M"
"%VENV_PY%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1
if errorlevel 1 (
    echo   [WARN] Existing .venv uses outdated/incompatible Python ^(!VENV_VER!^). OmniForge requires Python 3.12+.
    echo   [INFO] Automatically rebuilding .venv with Python %PY_VER%...
    echo [%DATE% %TIME%] Rebuilding incompatible .venv !VENV_VER! with Python %PY_VER% >> "%LOG_FILE%"
    set "RECREATE_VENV=1"
    goto :step3_recreate
)
if /i not "!VENV_MM!"=="%PY_MM%" (
    echo   [INFO] Existing .venv uses Python !VENV_VER! ^(detected system Python %PY_VER%^).
    echo   [INFO] Automatically upgrading .venv to Python %PY_VER%...
    echo [%DATE% %TIME%] Upgrading .venv !VENV_VER! to Python %PY_VER% >> "%LOG_FILE%"
    set "RECREATE_VENV=1"
    goto :step3_recreate
)
goto :step3_recreate

:step3_check_incomplete
if exist "%VENV_DIR%" (
    echo   [WARN] Incomplete .venv directory found without python.exe. Rebuilding...
    set "RECREATE_VENV=1"
)

:step3_recreate
if "%RECREATE_VENV%"=="1" (
    if exist "%VENV_DIR%" (
        echo   [INFO] Removing existing .venv directory...
        rmdir /s /q "%VENV_DIR%" >nul 2>&1
        if exist "%VENV_DIR%" (
            echo.
            echo [ERROR] Could not remove old "%VENV_DIR%" because a file is in use.
            echo         Please close any running OmniForge / Python windows, delete ".venv",
            echo         and run install.bat again.
            echo [%DATE% %TIME%] ERROR: Failed to remove locked .venv directory >> "%LOG_FILE%"
            goto :fail_exit
        )
    )
)

if not exist "%VENV_PY%" (
    echo   [INFO] Creating virtual environment at "%VENV_DIR%" with Python %PY_VER%...
    %SYS_PY_CMD% -m venv "%VENV_DIR%" >> "%LOG_FILE%" 2>&1
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment. Check "%LOG_FILE%".
        goto :fail_exit
    )
) else (
    echo   [OK] Existing compatible virtual environment detected. Reusing "%VENV_DIR%".
)

set "ACTIVE_VENV_VER=unknown"
for /f "delims=" %%V in ('.venv\Scripts\python.exe -c "import platform; print(platform.python_version())" 2^>nul') do set "ACTIVE_VENV_VER=%%V"
echo   [OK] Active .venv Python runtime: !ACTIVE_VENV_VER!
echo [%DATE% %TIME%] Active .venv Python: !ACTIVE_VENV_VER! >> "%LOG_FILE%"
"%VENV_PY%" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] Active .venv Python ^(!ACTIVE_VENV_VER!^) is older than Python 3.12.
    echo         Delete ".venv" manually and re-run install.bat.
    goto :fail_exit
)

set "PIP_NO_CACHE_DIR=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
echo   [INFO] Upgrading pip, setuptools, and wheel ^(no-cache mode to avoid Windows Defender / pip cache locks^)...
"%VENV_PY%" -m pip install --no-cache-dir --retries 5 --timeout 120 --upgrade pip setuptools wheel >> "%LOG_FILE%" 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] Failed to upgrade pip/setuptools/wheel. Check internet connection and "%LOG_FILE%".
    goto :fail_exit
)

:: ----------------------------------------------------------------------------
:: Step 4: Install Dependencies and OmniForge Package
:: ----------------------------------------------------------------------------
echo.
echo [Step 4/7] Installing OmniForge dependencies (%INSTALL_PROFILE%)...
set "PYYAML_FORCE_LIBYAML=0"
set "PIP_NO_CACHE_DIR=1"
set "PIP_FLAGS=--no-cache-dir --retries 5 --timeout 120"
"%VENV_PY%" -m pip uninstall -y omniforge >> "%LOG_FILE%" 2>&1
echo            This may take several minutes on first installation.

if /i "%INSTALL_PROFILE%"=="core" (
    "%VENV_PY%" -m pip install %PIP_FLAGS% --no-build-isolation --upgrade -e . >> "%LOG_FILE%" 2>&1
    if !errorlevel! neq 0 goto :pip_failed
) else if /i "%INSTALL_PROFILE%"=="full" (
    "%VENV_PY%" -m pip install %PIP_FLAGS% --upgrade -r "%ROOT_DIR%\requirements.txt" >> "%LOG_FILE%" 2>&1
    if !errorlevel! neq 0 goto :pip_failed
    "%VENV_PY%" -m pip install %PIP_FLAGS% --no-build-isolation --upgrade -e ".[api,ui,reporting,data,forecasting,deep,boosting,automl,responsible,nlp,vision,audio,build,dev]" >> "%LOG_FILE%" 2>&1
    if !errorlevel! neq 0 goto :pip_failed
) else (
    "%VENV_PY%" -m pip install %PIP_FLAGS% --upgrade -r "%ROOT_DIR%\requirements.txt" >> "%LOG_FILE%" 2>&1
    if !errorlevel! neq 0 goto :pip_failed
    "%VENV_PY%" -m pip install %PIP_FLAGS% --no-build-isolation --no-deps -e . >> "%LOG_FILE%" 2>&1
    if !errorlevel! neq 0 goto :pip_failed
)

"%VENV_PY%" -c "import omniforge.launcher, pandas, numpy" >> "%LOG_FILE%" 2>&1
if !errorlevel! neq 0 (
    echo [ERROR] OmniForge package import verification failed immediately after pip install.
    goto :pip_failed
)

echo   [INFO] Installing latest Windows .exe build prerequisites (PyInstaller, Pillow)...
"%VENV_PY%" -m pip install %PIP_FLAGS% --upgrade -r "%ROOT_DIR%\requirements-build.txt" >> "%LOG_FILE%" 2>&1
if !errorlevel! neq 0 goto :pip_failed

echo   [OK] All Python packages and OmniForge package installed cleanly.

:: ----------------------------------------------------------------------------
:: Step 5: Workspace Initialization (.env, directories, icon assets)
:: ----------------------------------------------------------------------------
echo.
echo [Step 5/7] Initializing environment configuration and icon assets...
if not exist "%ROOT_DIR%\.env" (
    if exist "%ROOT_DIR%\.env.example" (
        copy /y "%ROOT_DIR%\.env.example" "%ROOT_DIR%\.env" >nul
        echo   [OK] Created .env from .env.example
    )
) else (
    echo   [OK] Existing .env preserved.
)

if not exist "%ROOT_DIR%\data" mkdir "%ROOT_DIR%\data"
if not exist "%ROOT_DIR%\artifacts" mkdir "%ROOT_DIR%\artifacts"

"%VENV_PY%" "%ROOT_DIR%\packaging\generate_icon.py" >> "%LOG_FILE%" 2>&1
if !errorlevel! equ 0 (
    echo   [OK] Application icon verified at packaging\assets\omniforge.ico
) else (
    echo   [WARN] Could not regenerate omniforge.ico; continuing with existing assets if present.
)

:: ----------------------------------------------------------------------------
:: Step 6: Post-Install Verification
:: ----------------------------------------------------------------------------
echo.
echo [Step 6/7] Verifying OmniForge installation and analytical runtime...
"%VENV_PY%" -m omniforge.launcher --mode check
if !errorlevel! neq 0 (
    echo   [WARN] Self-check reported missing optional/runtime components. Review output above.
) else (
    echo   [OK] OmniForge self-check passed!
)

:: ----------------------------------------------------------------------------
:: Step 7: Build Standalone Windows .exe & Optional Desktop Shortcut
:: ----------------------------------------------------------------------------
echo.
echo [Step 7/7] Standalone Windows Executable (.exe) Generation...

if /i "%DO_BUILD_EXE%"=="ask" (
    if "%NO_PAUSE%"=="1" (
        set "DO_BUILD_EXE=no"
    ) else (
        set /p "BUILD_ANS=Build standalone Windows executable (OmniForge.exe) now? [Y/N] (default Y): "
        if "!BUILD_ANS!"=="" set "BUILD_ANS=Y"
        if /i "!BUILD_ANS!"=="Y" (
            set "DO_BUILD_EXE=yes"
            echo.
            echo   Select executable packaging mode:
            echo     [1] Directory Bundle - dist\OmniForge\OmniForge.exe ^(Fast startup, recommended^)
            echo     [2] Single File      - dist\OmniForge.exe     ^(Single portable executable^)
            set /p "MODE_ANS=Choose [1-2] (default 1): "
            if "!MODE_ANS!"=="2" (
                set "EXE_MODE=onefile"
            ) else (
                set "EXE_MODE=onedir"
            )
            set /p "CLI_ANS=Also build dedicated OmniForge-CLI.exe? [Y/N] (default N): "
            if /i "!CLI_ANS!"=="Y" set "EXE_TARGET=all"
        ) else (
            set "DO_BUILD_EXE=no"
        )
    )
)

if /i "%DO_BUILD_EXE%"=="yes" (
    echo.
    echo   [BUILD] Compiling Windows executable ^(mode=%EXE_MODE%, target=%EXE_TARGET%^)...
    set "INNO_FLAG="
    if "%DO_INNO%"=="1" set "INNO_FLAG=--with-inno"
    "%VENV_PY%" "%ROOT_DIR%\packaging\build_exe.py" --mode %EXE_MODE% --target %EXE_TARGET% !INNO_FLAG!
    if !errorlevel! neq 0 (
        echo [ERROR] Executable build failed. See output above and "%LOG_FILE%".
        goto :fail_exit
    )
    echo   [OK] Standalone executable built successfully in "%ROOT_DIR%\dist\"!
)

if /i "%DO_SHORTCUT%"=="ask" (
    if "%NO_PAUSE%"=="1" (
        set "DO_SHORTCUT=no"
    ) else (
        echo.
        set /p "SHORTCUT_ANS=Create a Windows Desktop shortcut for OmniForge? [Y/N] (default Y): "
        if "!SHORTCUT_ANS!"=="" set "SHORTCUT_ANS=Y"
        if /i "!SHORTCUT_ANS!"=="Y" (
            set "DO_SHORTCUT=yes"
        ) else (
            set "DO_SHORTCUT=no"
        )
    )
)

if /i "%DO_SHORTCUT%"=="yes" (
    set "SHORTCUT_TARGET=%ROOT_DIR%\start.bat"
    if exist "%ROOT_DIR%\dist\OmniForge\OmniForge.exe" set "SHORTCUT_TARGET=%ROOT_DIR%\dist\OmniForge\OmniForge.exe"
    if exist "%ROOT_DIR%\dist\OmniForge.exe" set "SHORTCUT_TARGET=%ROOT_DIR%\dist\OmniForge.exe"
    set "ICON_FILE=%ROOT_DIR%\packaging\assets\omniforge.ico"
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$ws = New-Object -ComObject WScript.Shell; $sc = $ws.CreateShortcut([Environment]::GetFolderPath('Desktop') + '\OmniForge.lnk'); $sc.TargetPath = '!SHORTCUT_TARGET!'; $sc.WorkingDirectory = '%ROOT_DIR%'; if (Test-Path '!ICON_FILE!') { $sc.IconLocation = '!ICON_FILE!' }; $sc.Description = 'OmniForge Enterprise'; $sc.Save()" >nul 2>&1
    if !errorlevel! equ 0 (
        echo   [OK] Desktop shortcut created pointing to: !SHORTCUT_TARGET!
    ) else (
        echo   [WARN] Could not create Desktop shortcut automatically.
    )
)

echo.
echo ============================================================================
echo   Installation Complete!
echo ============================================================================
echo   Start Platform   : Double-click start.bat
echo   Build/Rebuild EXE: Double-click build_exe.bat
if exist "%ROOT_DIR%\dist\OmniForge\OmniForge.exe" echo   Desktop EXE      : %ROOT_DIR%\dist\OmniForge\OmniForge.exe
if exist "%ROOT_DIR%\dist\OmniForge.exe"     echo   Portable EXE     : %ROOT_DIR%\dist\OmniForge.exe
if exist "%ROOT_DIR%\dist\OmniForge-CLI.exe" echo   CLI EXE          : %ROOT_DIR%\dist\OmniForge-CLI.exe
echo   Install Log      : %LOG_FILE%
echo ============================================================================
echo [%DATE% %TIME%] === Installation Finished Successfully === >> "%LOG_FILE%"

popd
if "%NO_PAUSE%"=="0" pause
exit /b 0

:pip_failed
echo.
echo [ERROR] Dependency installation via pip failed.
echo         Last 20 lines of "%LOG_FILE%":
echo ----------------------------------------------------------------------------
powershell -NoProfile -Command "if (Test-Path '%LOG_FILE%') { Get-Content -Path '%LOG_FILE%' -Tail 20 }" 2>nul
echo ----------------------------------------------------------------------------
echo         Full install log: "%LOG_FILE%"
echo.
echo   [TIP] If you saw "[Errno 13] Permission denied" on a wheel or .venv file:
echo         1. Close any running Python / OmniForge windows and clear stale pip cache:
echo              .venv\Scripts\python.exe -m pip cache purge
echo         2. Temporarily allow your antivirus / Windows Defender exclusion for this folder.
echo         3. Re-run install.bat ^(now uses --no-cache-dir automatically^).
goto :fail_exit

:fail_exit
echo.
echo [FAILED] Installation did not complete. Review messages above and "%LOG_FILE%".
popd
if "%NO_PAUSE%"=="0" pause
exit /b 1

@echo off
REM ============================================================================
REM  OmniForge v2.0.0 - Bootstrap Installer Builder (Windows)
REM
REM  Produces:  dist\installer\OmniForge-2.0.0-Setup.exe   (~30-35 MB)
REM
REM  This replaces the PyInstaller flow for the installer:
REM      app code + embeddable CPython + uv + hash-pinned locks
REM  The Python dependencies are installed into a private venv ON THE USER'S PC
REM  by installer\provision.py (that is what keeps the download small).
REM
REM  Usage:
REM      build_installer.bat                 full build (locks -> payload -> setup)
REM      build_installer.bat --skip-locks    reuse the committed locks\*.lock
REM      build_installer.bat --clean         wipe dist\omniforge-bootstrap first
REM      build_installer.bat --check         only verify the toolchain, build nothing
REM      build_installer.bat --launch-mode exe   ship a PyInstaller build instead
REM ============================================================================
setlocal EnableDelayedExpansion
cd /d "%~dp0"

set "ROOT=%CD%"
set "PY_EXE=python"
set "ISCC="
set "PY_VERSION=3.13.9"
set "LAUNCH_MODE=py"
set "SKIP_LOCKS=0"
set "DO_CLEAN=0"
set "CHECK_ONLY=0"
set "PAUSE_AT_END=1"

:parse
if "%~1"=="" goto after_parse
if /I "%~1"=="--skip-locks"  set "SKIP_LOCKS=1" & shift & goto parse
if /I "%~1"=="--clean"       set "DO_CLEAN=1"   & shift & goto parse
if /I "%~1"=="--check"       set "CHECK_ONLY=1" & shift & goto parse
if /I "%~1"=="--no-pause"    set "PAUSE_AT_END=0" & shift & goto parse
if /I "%~1"=="--launch-mode" set "LAUNCH_MODE=%~2" & shift & shift & goto parse
echo [WARN] unknown option: %~1
shift
goto parse
:after_parse

echo.
echo ============================================================================
echo   OmniForge Bootstrap Installer Builder
echo ============================================================================

REM ---------------------------------------------------------------- toolchain
where %PY_EXE% >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found on PATH. Install Python 3.12+ and re-run.
    goto fail
)
for /f "delims=" %%v in ('%PY_EXE% -c "import sys;print('%%d.%%d'%%sys.version_info[:2])"') do set "PYVER=%%v"
echo [1/6] Python detected: !PYVER!

if exist "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if exist "%ProgramFiles%\Inno Setup 6\ISCC.exe"       set "ISCC=%ProgramFiles%\Inno Setup 6\ISCC.exe"
if not defined ISCC (
    echo [ERROR] Inno Setup 6 not found.
    echo         Install it from https://jrsoftware.org/isdl.php  ^(or: winget install JRSoftware.InnoSetup^)
    echo         Then re-run this script.
    goto fail
)
echo         Inno Setup: "!ISCC!"
set "ISCC_PATH=!ISCC!"
REM ArchitecturesAllowed/ArchitecturesInstallIn64BitMode accept "x64compatible" only in
REM Inno Setup 6.3+. On an older 6.x ISCC fails with "unknown directive value", which is a
REM confusing message; check the version up front instead. Version detection is best effort.
set "INNO_VER="
for /f "usebackq delims=" %%v in (`powershell -NoProfile -Command "(Get-Item -LiteralPath $env:ISCC_PATH).VersionInfo.FileVersion" 2^>nul`) do set "INNO_VER=%%v"
if not defined INNO_VER (
    echo [WARN] Could not read the Inno Setup version; skipping the 6.3 check.
) else (
    for /f "tokens=1,2 delims=." %%a in ("!INNO_VER!") do (
        set "INNO_MAJOR=%%a"
        set "INNO_MINOR=%%b"
    )
    set "INNO_OK=0"
    if !INNO_MAJOR! GTR 6 set "INNO_OK=1"
    if !INNO_MAJOR! EQU 6 if !INNO_MINOR! GEQ 3 set "INNO_OK=1"
    if "!INNO_OK!"=="0" (
        echo [ERROR] Inno Setup !INNO_VER! is too old; version 6.3 or newer is required
        echo         ^("x64compatible" architecture identifiers do not exist before 6.3^).
        echo         Download: https://jrsoftware.org/isdl.php
        goto fail
    )
    echo         Inno Setup version: !INNO_VER! ^(>= 6.3: ok^)
)

%PY_EXE% -m pip install --quiet --upgrade uv
if errorlevel 1 (
    echo [ERROR] Could not install/upgrade 'uv'. Check your network or run: pip install uv
    goto fail
)
where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 'uv' is not on PATH after installation. Run: python -m pip install uv
    goto fail
)
echo         uv: ok

if "%CHECK_ONLY%"=="1" (
    echo.
    echo [OK] Toolchain looks complete. Nothing was built ^(--check^).
    goto done
)

REM ------------------------------------------------------------ launcher images
echo.
echo [2/6] Preparing CPython launcher images ^(venvlauncher.exe^)...
if not exist ".cache\bootstrap" mkdir ".cache\bootstrap"
set "LAUNCHER_ZIP=%ROOT%\.cache\bootstrap\python-%PY_VERSION%-amd64.zip"
if not exist "!LAUNCHER_ZIP!" (
    echo         Downloading python-%PY_VERSION%-amd64.zip ^(~33 MB, one time^)...
    %PY_EXE% -c "import urllib.request,sys;urllib.request.urlretrieve('https://www.python.org/ftp/python/%PY_VERSION%/python-%PY_VERSION%-amd64.zip', sys.argv[1])" "!LAUNCHER_ZIP!"
    if errorlevel 1 (
        echo [WARN] Download failed. The installer will fall back to OmniForge.cmd instead of OmniForge.exe.
        set "LAUNCHER_ZIP="
    )
) else (
    echo         Reusing cached !LAUNCHER_ZIP!
)

REM ------------------------------------------------------------------- locks
echo.
echo [3/6] Hash-pinned locks...
if "%SKIP_LOCKS%"=="1" (
    echo         Skipped ^(--skip-locks^) - using the committed locks\*.lock
) else (
    %PY_EXE% scripts\build_locks.py
    if errorlevel 1 (
        echo [ERROR] Lock generation failed.
        goto fail
    )
)

REM ----------------------------------------------------------------- payload
echo.
echo [4/6] Staging the bootstrap payload...
if "%DO_CLEAN%"=="1" if exist "dist\omniforge-bootstrap" (
    echo         Removing old dist\omniforge-bootstrap
    rmdir /S /Q "dist\omniforge-bootstrap"
)
REM Incremental: keep a complete app\ tree, refresh runtime/locks/launchers only.
REM --clean forces a full re-copy of app\ as well.
set "STAGE_FLAGS=--skip-staged"
if "%DO_CLEAN%"=="1" set "STAGE_FLAGS="
if defined LAUNCHER_ZIP (
    %PY_EXE% scripts\build_bootstrap.py --launcher-zip "!LAUNCHER_ZIP!" %STAGE_FLAGS%
) else (
    %PY_EXE% scripts\build_bootstrap.py %STAGE_FLAGS%
)
if errorlevel 1 (
    echo [ERROR] Payload staging failed. See the messages above.
    goto fail
)

REM Every file the installer references must exist in the payload. LICENSE.txt is the one
REM ISCC actually reads ([Setup] LicenseFile only accepts .txt/.rtf); LICENSE is the original.
set "PAYLOAD_MISSING=0"
for %%F in ("app\LICENSE" "app\LICENSE.txt" "app\omniforge.py" "app\installer\provision.py" "runtime\python\python.exe" "runtime\python\OmniForge.exe" "locks\advanced.lock") do (
    if not exist "dist\omniforge-bootstrap\%%~F" (
        echo [ERROR] Payload is missing %%~F
        set "PAYLOAD_MISSING=1"
    )
)
if "!PAYLOAD_MISSING!"=="1" (
    echo         Re-run with --clean:  build_installer.bat --clean
    goto fail
)
echo         Payload self-check: all required files present

REM ------------------------------------------------------- validate the .iss
echo.
echo [5a/6] Validating the Inno Setup script...
%PY_EXE% scripts\check_iss.py --stage dist\omniforge-bootstrap
if errorlevel 1 (
    echo [ERROR] The .iss failed validation. Fix the reported lines before compiling.
    echo         Run: python scripts\check_iss.py --stage dist\omniforge-bootstrap
    goto fail
)

REM ------------------------------------------------------------------ compile
echo.
echo [5b/6] Compiling the Setup wizard (LaunchMode=%LAUNCH_MODE%)...
if /I "%LAUNCH_MODE%"=="exe" (
    if not exist "dist\OmniForge\OmniForge.exe" (
        echo [ERROR] LaunchMode=exe needs a PyInstaller build first:
        echo         build_exe.bat --onedir
        goto fail
    )
    "!ISCC!" /DLaunchMode=exe "installer\omniforge_installer_tiers.iss"
) else (
    "!ISCC!" "installer\omniforge_installer_tiers.iss"
)
if errorlevel 1 (
    echo [ERROR] ISCC failed. The message above tells you which file is missing.
    goto fail
)

echo.
echo [6/6] Result:
for %%f in ("dist\installer\*.exe") do (
    for %%s in ("%%~ff") do set /a "MB=%%~zs/1048576"
    echo         %%~nxf   ^(!MB! MB^)
)
echo.
echo ============================================================================
echo   [SUCCESS] Installer written to: %ROOT%\dist\installer\
echo   Next: test on a clean VM, then sign the exe before distribution.
echo ============================================================================
goto done

:fail
echo.
echo ============================================================================
echo   [FAILED] Build aborted. Nothing was installed; dist\ may be incomplete.
echo   Tip: run the steps manually to see full output:
echo     python scripts\build_locks.py
echo     python scripts\build_bootstrap.py --launcher-zip .cache\bootstrap\python-%PY_VERSION%-amd64.zip
echo     "%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe" installer\omniforge_installer_tiers.iss
echo ============================================================================
if "%PAUSE_AT_END%"=="1" pause
exit /b 1

:done
if "%PAUSE_AT_END%"=="1" pause
exit /b 0

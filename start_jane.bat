@echo off
setlocal
chcp 65001 >nul
title Jane - local data analyst
rem Jane - local AI data analyst (made by Kumar Harsh)
rem DEFAULT launcher: the React app served by the bridge on http://localhost:8100
rem (one process, no dev server needed). The old Streamlit UI lives in
rem start_jane_streamlit.bat. Run "start_jane.bat --banner" to just print the logo.
cd /d "%~dp0"

if "%~1"=="--banner" (
    call :banner
    exit /b 0
)

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Python venv not found. Create it first:
    echo   py -3 -m venv .venv
    echo   .venv\Scripts\python -m pip install -e .
    pause
    exit /b 1
)

rem The React UI is not shipped with this repo. Build it only if a local
rem frontend/ was dropped in; otherwise serve the API on its own.
if exist "frontend\package.json" if not exist "frontend\dist\index.html" (
    echo.
    echo Building the local frontend - first run only, this takes a minute...
    pushd frontend
    call npm install
    call npm run build
    popd
)
if not exist "frontend\dist\index.html" (
    echo.
    echo No built UI found - starting the API only.
    echo The React frontend is private; run start_jane_streamlit.bat for the bundled UI.
)

call :banner

rem If Jane already answers on :8100, do not start a second bridge - just browse.
powershell -NoProfile -Command "if (Get-NetTCPConnection -LocalPort 8100 -State Listen -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }" >nul 2>&1
if not errorlevel 1 (
    echo.
    echo Jane is already running
    if exist "frontend\dist\index.html" (
        start "" http://localhost:8100
    ) else (
        start "" http://localhost:8100/docs
    )
    timeout /t 3 /nobreak >nul
    exit /b 0
)

echo.
echo Starting Jane on http://localhost:8100 ...
start "Jane bridge" /min ".venv\Scripts\python.exe" -m uvicorn server:app --port 8100

rem Wait until the bridge answers /api/health (up to ~30s), then open the browser.
powershell -NoProfile -Command "$u='http://127.0.0.1:8100/api/health'; for($i=0; $i -lt 60; $i++){ try { Invoke-WebRequest -UseBasicParsing -Uri $u -TimeoutSec 2 | Out-Null; exit 0 } catch { Start-Sleep -Milliseconds 500 } }; exit 1" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Jane did not come up within 30 seconds - check the "Jane bridge" window for the error.
    pause
    exit /b 1
)

if exist "frontend\dist\index.html" (
    start "" http://localhost:8100
    echo.
    echo Jane is running. Close the "Jane bridge" window to stop it.
) else (
    start "" http://localhost:8100/docs
    echo.
    echo Jane API is running - interactive docs at http://localhost:8100/docs
    echo The web UI is private; use start_jane_streamlit.bat for the bundled one.
)
pause
exit /b 0

:banner
echo(
echo    ██   ▄███▄   █▄  █   █▀▀▀▀
echo    ██   █   █   █▀▄ █   █
echo    ██   █▀▀▀█   █ ▀▄█   █▀▀▀
echo █  ██   █   █   █  ▀█   █
echo ▀███▀   █   █   █   █   █▄▄▄▄
echo(
echo    local AI data analyst  ·  http://localhost:8100
echo    made by Kumar Harsh
echo(
goto :eof

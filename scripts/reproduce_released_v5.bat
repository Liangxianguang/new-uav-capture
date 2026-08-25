@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0reproduce_released_v5.ps1" %*
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%

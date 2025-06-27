@echo off
setlocal EnableDelayedExpansion

echo =====================================
echo        Pixi Uninstaller Script
echo =====================================

REM Single Confirmation
set /p CONFIRM=Are you sure you want to uninstall Pixi and delete all environments? (y/yes): 
set "CONFIRM=!CONFIRM:~0,1!"
if /I not "!CONFIRM!"=="y" (
    echo Uninstallation aborted.
    pause
    exit /b 1
)

REM Define install path
set "PIXIPATH=%USERPROFILE%\.pixi"

REM Check if Pixi is installed
if not exist "%PIXIPATH%" (
    echo Pixi does not appear to be installed at %PIXIPATH%.
    pause
    exit /b 0
)

REM Remove Pixi from User PATH
echo Removing Pixi from PATH...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$userPath = [Environment]::GetEnvironmentVariable('Path', 'User');" ^
    "$newPath = ($userPath -split ';' | Where-Object { $_ -notmatch '\\.pixi\\bin' }) -join ';';" ^
    "[Environment]::SetEnvironmentVariable('Path', $newPath, 'User')"

REM Delete Pixi files
echo Deleting %PIXIPATH% ...
rmdir /s /q "%PIXIPATH%"

REM Optional: Clear Pixi cached environments
set "CACHE=%LOCALAPPDATA%\pixi"
if exist "%CACHE%" (
    echo Deleting Pixi cache at %CACHE% ...
    rmdir /s /q "%CACHE%"
)

echo Pixi has been uninstalled.

echo Removing shortcut from Desktop

set "SHORTCUT_NAME=Launch Napari Cool Tools.lnk"
set "SHORTCUT_PATH=%USERPROFILE%\Desktop\%SHORTCUT_NAME%"

if exist "%SHORTCUT_PATH%" (
    del "%SHORTCUT_PATH%"
    echo Shortcut removed: %SHORTCUT_PATH%
) else (
    echo Shortcut not found: %SHORTCUT_PATH%
)

echo You can now delete this folder (including this uninstall.bat file).

pause
endlocal

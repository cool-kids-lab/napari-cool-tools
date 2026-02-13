@echo off
setlocal

echo =====================================
echo      Napari-Cool-Tools Installer
echo =====================================


REM Check if pixi is in PATH
where pixi >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Pixi not found. Installing...

    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "iwr https://pixi.sh/install.ps1 -UseBasicParsing | iex"

    REM Set Pixi path for current session
    set "PIXIPATH=%USERPROFILE%\.pixi\bin"
    set "PATH=%PIXIPATH%;%PATH%"

    REM Permanently add Pixi to user PATH if not already present
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
        "$currentPath = [Environment]::GetEnvironmentVariable('Path', 'User');" ^
        "if (-not $currentPath.Contains('.pixi\\bin')) {" ^
        "    [Environment]::SetEnvironmentVariable('Path', $currentPath + ';%PIXIPATH%', 'User')" ^
        "}"

    echo Pixi installed and added to user PATH.

) else (
    echo Pixi is already installed.
)

"%USERPROFILE%\.pixi\bin\pixi.exe" reinstall
"%USERPROFILE%\.pixi\bin\pixi.exe" run python -m cool_tools_install_script

REM Creating shortcut to Desktop
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$desktop = [Environment]::GetFolderPath('Desktop');" ^
    "$shortcut = (New-Object -COM WScript.Shell).CreateShortcut($desktop + '\Launch Napari Cool Tools.lnk');" ^
    "$shortcut.TargetPath = '%CD%\launch_cool-tools_pixi.bat';" ^
    "$shortcut.WorkingDirectory = '%CD%';" ^
    "$shortcut.WindowStyle = 1;" ^
    "$shortcut.IconLocation = '%CD%\napari.ico';" ^
    "$shortcut.Save()"

echo -------------------------------------
echo NOTE: Package installed to your USER environment path and a shortcut was created on your Desktop.
echo       This does NOT require admin privileges.
echo       It is only available for your Windows user account.
echo -------------------------------------
echo.

echo Package installation complete.
pause
endlocal
@echo off
setlocal
SETLOCAL ENABLEDELAYEDEXPANSION

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
    REM set "PIXI_HOME=%USERPROFILE%\.pixi"
    REM set "PIXIPATH=%USERPROFILE%\.pixi\bin"

    for /f "delims=" %%a in ('where pixi') do (
        REM echo %%a
        set temp=%%a
        REM echo The path to the pixi binary is: !temp!
    )

    set PIXI_PATH=!temp!

    if defined PIXI_PATH (
        echo The path to the pixi binary is: !PIXI_PATH!
        set "PATH=%PIXI_PATH%;%PATH%"

        REM Permanently add Pixi to user PATH if not already present
        powershell -NoProfile -ExecutionPolicy Bypass -Command ^
            "$currentPath = [Environment]::GetEnvironmentVariable('Path', 'User');" ^
            "if (-not $currentPath.Contains('.pixi\\bin')) {" ^
            "    [Environment]::SetEnvironmentVariable('Path', $currentPath + ';%PIXI_PATH%', 'User')" ^
            "}"

        echo Pixi installed and added to user PATH.

    ) else (
        echo Pixi powershell install failed.
    )

) else (
    echo Pixi is already installed.
    REM echo %PIXI_HOME%
    REM WHERE pixi
    
    for /f "delims=" %%a in ('where pixi') do (
        REM echo %%a
        set temp=%%a
        REM echo The path to the pixi binary is: !temp!
    )

    set PIXI_PATH=!temp!

    if defined PIXI_PATH (
        echo The path to the pixi binary is: !PIXI_PATH!
    )
)

"%PIXI_PATH%" reinstall
"%PIXI_PATH%" run python -m cool_tools_install_script --pixi_binary_path "%PIXI_PATH%"

GOTO EndCommentBlock

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

:EndCommentBlock

pause
endlocal

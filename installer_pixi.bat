@echo off

:start
cls

call pixi reinstall
call pixi run python -m cool_tools_install_script

echo: &echo Installation is complete !! &echo:
call cmd /k
@pause
@exit
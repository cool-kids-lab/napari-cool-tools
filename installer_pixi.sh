!#/bin/bash

pwd
ls
echo Confirm correct directory

pixi reinstall
pixi run python -m cool_tools_install_script

echo instalation complete

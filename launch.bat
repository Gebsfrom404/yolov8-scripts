@echo off
set VENV_PATH=ultralytics/ultralytics-venv

echo Activating virtual environment...
call "%VENV_PATH%\Scripts\activate"
echo Virtual environment activated.

:STARTMENU
if defined finished_script echo %finished_script%
echo 1. Train model
echo 2. Generate masks for simple model
echo 3. Generate masks for obb model
echo 8. Just venv terminal
echo 9. exit
CHOICE /N /C 12389

IF %ERRORLEVEL% == 1 GOTO TRAIN
IF %ERRORLEVEL% == 2 GOTO GENERATE
IF %ERRORLEVEL% == 3 GOTO GENERATEOBB
IF %ERRORLEVEL% == 4 GOTO JUSTVENV
IF %ERRORLEVEL% == 5 GOTO END


:TRAIN
setlocal
set /p out_name=Enter output name:
echo Specify model type, 1-ob, 2-obb, 3-segm
CHOICE /N /C 123

echo Training %out_name% model --model_type %ERRORLEVEL%
train.py --output_name %out_name% --model_type %ERRORLEVEL%
set finished_script="Finished training model: %out_name%"
GOTO STARTMENU

:GENERATE
set /p input_path=Enter path to folder with images:
set /p model_name=Enter model name to use:
generate.py "%input_path%" "%model_name%"
set finished_script="Finished generating masks"
GOTO STARTMENU

:GENERATEOBB
setlocal
generate_obb.py
set finished_script="Finished generating masks"
GOTO STARTMENU

:JUSTVENV
cmd /k

:END
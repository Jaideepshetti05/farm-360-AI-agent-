@echo off
echo [1/2] running dairy_module.download_all_datasets
venv\Scripts\python -m dairy_module.download_all_datasets
if %errorlevel% neq 0 exit /b %errorlevel%

echo [2/2] running dairy_module.train_all_models
venv\Scripts\python -m dairy_module.train_all_models

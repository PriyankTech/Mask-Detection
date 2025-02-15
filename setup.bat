@echo off
echo ===============================
echo Creating Virtual Environment...
echo ===============================
python -m venv venv

echo ===============================
echo Activating Virtual Environment...
echo ===============================
call venv\Scripts\activate

echo ===============================
echo Upgrading pip...
echo ===============================
python -m pip install --upgrade pip

echo ===============================
echo Installing Required Modules...
echo ===============================
pip install numpy opencv-python tensorflow tabulate tqdm Pillow scipy matplotlib pandas scikit-learn h5py

echo ===============================
echo Saving Requirements to requirements.txt...
echo ===============================
pip freeze > requirements.txt

echo ===============================
echo Setup Complete! Virtual Environment is Ready.
echo ===============================
pause

@echo on
call C:\users\instant\Anaconda3\Scripts\activate.bat
call conda activate melt
cd c:\tutorial\onnx
python 98kp_2_onnx.py
pause
conda deactivate
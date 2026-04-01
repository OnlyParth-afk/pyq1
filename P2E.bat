@echo off
title P2E — NEET PG MCQ Bank
cd /d "D:\Study meteial\NEET PG QBANK\p2e"

echo.
echo  ======================================
echo    P2E — NEET PG MCQ Bank
echo  ======================================
echo.

:: Run loader silently to pick up any new splits
echo  [1/2] Loading new questions...
python loader.py --stem "5.Micro PYQ (2017-2022)" >loader_log.txt 2>&1
echo  Done.

:: Start the server
echo  [2/2] Starting server...
echo.
echo  Opening: http://127.0.0.1:5000
echo  Press Ctrl+C to stop
echo.

python run.py --skip-extract
pause

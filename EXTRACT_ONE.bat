@echo off
cd /d "D:\Study meteial\NEET PG QBANK\p2e"
echo ============================================
echo  P2E -- Extract Single PDF
echo  Drag a PDF file onto this .bat to use it
echo ============================================
if "%~1"=="" (
    echo ERROR: Drag a PDF file onto this .bat file
    pause
    exit /b
)
python extract.py --pdf "%~1" --out "D:\Study meteial\NEET PG QBANK\p2e\output" --dpi 300
pause

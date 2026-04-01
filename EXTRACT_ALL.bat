@echo off
cd /d "D:\Study meteial\NEET PG QBANK\p2e"
echo ============================================
echo  P2E -- Extract All PDFs (300 DPI)
echo ============================================
python extract.py --all --pdfs "D:\Study meteial\NEET PG QBANK\pyq" --out "D:\Study meteial\NEET PG QBANK\p2e\output" --dpi 300
pause

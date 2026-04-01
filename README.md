# P2E - NEET PG MCQ Extraction & Study Platform

## Quick Start

### 1. First time setup
    python setup_folders.py
    pip install -r requirements.txt

### 2. Extract all PDFs
    Double-click: EXTRACT_ALL.bat
    or: python extract.py --all --pdfs pyq/ --out output/

### 3. Extract single PDF
    Drag PDF onto: EXTRACT_ONE.bat
    or: python extract.py --pdf file.pdf --out output/

### 4. Start web server
    Double-click: START_WEBONLY.bat
    or: python run.py --skip-extract

### 5. Full pipeline (extract + web)
    Double-click: START.bat
    or: python run.py

## Web UI

    http://127.0.0.1:5000

| URL               | Purpose                  |
|-------------------|--------------------------|
| /                 | Dashboard                |
| /questions        | Browse all questions      |
| /study            | Start study session       |
| /session          | Live exam/practice        |
| /analytics        | FSRS + accuracy charts    |
| /topics           | Topic browser             |
| /review           | PDF review dashboard      |
| /review/<pdf>     | Question list for one PDF |

## Folder Structure

    p2e/
    +-- output/
    |   +-- index.json              # PDF index
    |   +-- marrow.db               # Questions DB
    |   +-- progress.db             # User progress DB
    |   +-- 1.Anat PYQ (2017-2022)/
    |   |   +-- raw.json            # Extracted data
    |   |   +-- clean.json          # After your edits
    |   |   +-- manifest.json       # Metadata + edit log
    |   |   +-- images/             # 300 DPI PNGs
    |   +-- ...
    +-- templates/                  # Flask HTML
    +-- static/                     # CSS, JS
    +-- logs/                       # Run logs
    +-- *.py                        # Source files

## Audit Mode

All editing is LOCKED by default.
Click 'Enable Audit Mode' on the Review page to unlock editing.
Resets automatically when browser tab closes.
Every change is logged in manifest.json edit_log.

## Image Quality

All images extracted at 300 DPI (2x vs old 150 DPI).
PNG compression=1 (near-lossless).
Small images auto-upscaled with INTER_CUBIC.
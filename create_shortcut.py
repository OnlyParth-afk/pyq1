"""
create_shortcut.py — Run once to create P2E desktop shortcut
python create_shortcut.py
"""
import os, sys, winreg

def create_shortcut():
    try:
        import winshell
        from win32com.client import Dispatch
    except ImportError:
        print("Installing required packages...")
        os.system("pip install winshell pywin32 --quiet")
        import winshell
        from win32com.client import Dispatch

    desktop    = winshell.desktop()
    bat_path   = r"D:\Study meteial\NEET PG QBANK\p2e\P2E.bat"
    icon_path  = r"D:\Study meteial\NEET PG QBANK\p2e\p2e_icon.ico"
    link_path  = os.path.join(desktop, "P2E Study.lnk")

    shell = Dispatch("WScript.Shell")
    shortcut = shell.CreateShortCut(link_path)
    shortcut.Targetpath      = bat_path
    shortcut.WorkingDirectory= r"D:\Study meteial\NEET PG QBANK\p2e"
    shortcut.Description     = "P2E NEET PG MCQ Bank"
    shortcut.WindowStyle     = 1   # normal window
    if os.path.exists(icon_path):
        shortcut.IconLocation = icon_path
    shortcut.save()

    print(f"Desktop shortcut created: {link_path}")
    print("Double-click 'P2E Study' on your desktop to launch!")

if __name__ == "__main__":
    create_shortcut()

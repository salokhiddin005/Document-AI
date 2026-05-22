' Run the bot silently in background (no console window)
Set WShell = CreateObject("WScript.Shell")
WShell.Run Chr(34) & "c:\Users\OEM\OneDrive\Desktop\Custom OCR and Document AI System for Handwritten Record Processing\run_bot.bat" & Chr(34), 0, False

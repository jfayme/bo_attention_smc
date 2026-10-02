@echo off
rem Stop every queue now: kills each driver with its campaign processes.
rem Campaigns already written stay; the running ones are lost and rerun on start.
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*bo_attention_smc.runs.queue*' -and $_.Name -eq 'python.exe' } | ForEach-Object { taskkill /T /F /PID $_.ProcessId }"

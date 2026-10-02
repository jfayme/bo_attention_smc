@echo off
rem Start (or resume) the queues runs\queue\queue_smc.txt and queue_map.txt:
rem one minimised driver per queue, skipped if that queue is already running.
rem Its output goes to runs\queue\driver_<queue>.log.
set PYTHON=D:\Conda\envs\aimnet-bo\python.exe
cd /d %~dp0..\..
if exist runs\queue\STOP del runs\queue\STOP
for %%q in (smc map) do (
  powershell -NoProfile -Command "if (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*runs.queue runs/queue/queue_%%q.txt*' -and $_.Name -eq 'python.exe' }) { exit 1 } else { exit 0 }" && start "queue_%%q" /min cmd /c "%PYTHON% -u -m bo_attention_smc.runs.queue runs/queue/queue_%%q.txt >> runs\queue\driver_%%q.log 2>&1"
)

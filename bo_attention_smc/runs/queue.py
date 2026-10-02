"""Run the blocks of a queue file one after another; stop and resume at will.

    python -m bo_attention_smc.runs.queue runs/queue/queue_smc.txt

Each line of the queue that is not empty or a comment is `label | module
args...`. The first block not yet listed in <queue>.done runs as
`python -u -m module args`, its output appended to <label>.log beside the
queue. A block that exits 0 is marked done and the results table beside the
queue is rebuilt; one that fails is marked failed and skipped. The queue is
re-read before every block, so lines can be added or reordered while it
runs. The campaign runner skips campaigns already in its run folder, so a
killed block resumes where it stopped.

Start or resume the two queues runs/queue/queue_smc.txt and
queue_map.txt, with bo_attention_smc/runs/start.cmd. Stop: create a file
STOP beside the queue (it ends after the current block), or run stop.cmd
(it kills every queue and its campaigns now).

"""

import subprocess
import sys
import time
from pathlib import Path


def blocks(queue: Path) -> list[tuple[str, list[str]]]:
    """Read the queue's blocks: each label, and its module with arguments."""
    found = []
    for line in queue.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        label, command = (part.strip() for part in line.split("|", 1))
        found.append((label, command.split()))
    return found


def main() -> None:
    """Run the queue's blocks until it is finished or asked to stop."""
    queue = Path(sys.argv[1])
    folder = queue.parent
    done_file = queue.with_suffix(".done")
    while True:
        if (folder / "STOP").exists():
            print("STOP file found; ending", flush=True)
            return
        done = set()
        if done_file.is_file():
            for line in done_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    done.add(line.split()[0])
        todo = [block for block in blocks(queue) if block[0] not in done]
        if not todo:
            print("queue finished", flush=True)
            return
        label, cmd = todo[0]
        now = time.strftime("%H:%M:%S")
        print(f"{now} start {label}: {' '.join(cmd)}", flush=True)
        with (folder / f"{label}.log").open("a", encoding="utf-8") as log:
            code = subprocess.call(
                [sys.executable, "-u", "-m", *cmd],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        status = "ok" if code == 0 else f"failed({code})"
        with done_file.open("a", encoding="utf-8") as handle:
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            handle.write(f"{label} {status} {now}\n")
        print(f"{time.strftime('%H:%M:%S')} end {label}: {status}", flush=True)
        # the table is a convenience: a failure to rebuild it stops nothing
        subprocess.call(
            [
                sys.executable,
                "-m",
                "bo_attention_smc.diagnostics_and_metrics.report",
                "table",
                str(folder),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


if __name__ == "__main__":
    main()

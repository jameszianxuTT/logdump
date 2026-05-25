#!/usr/bin/env python3
# pip install psutil matplotlib
# python memory_logger.py --pid 12345 --interval 0.1 --csv memory.csv --png memory.png
# python memory_logger.py --name worker --interval 0.1 --csv memory.csv --png memory.png
# python memory_logger.py --from-csv memory.csv --png memory.png
# python memory_logger.py --top 10 --interval 0.5 --csv top_memory.csv --png top_memory.png

import argparse
import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import psutil


def bytes_to_mib(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def get_oom_score(pid: int) -> int | None:
    """Read OOM score from /proc/[pid]/oom_score (Linux only, range 0-1000)."""
    try:
        with open(f"/proc/{pid}/oom_score") as f:
            return int(f.read().strip())
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, OSError):
        return None


def get_swap_bytes(pid: int) -> int:
    """Read VmSwap from /proc/[pid]/status and return bytes (Linux only)."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmSwap:"):
                    # VmSwap is reported as: "VmSwap:\t<value> kB"
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
                    break
    except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError, OSError):
        pass
    return 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Log RSS, swap, and OOM score for a specific PID and plot it, or render a PNG from an existing CSV."
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--pid", type=int, help="Target process ID")
    target_group.add_argument(
        "--name",
        type=str,
        help="Attach to first process whose name exactly matches this string",
    )
    target_group.add_argument(
        "--from-csv",
        type=Path,
        help="Read an existing CSV and generate a PNG without attaching to a process",
    )
    target_group.add_argument(
        "--top",
        type=int,
        metavar="N",
        help="Monitor top N memory-consuming processes system-wide",
    )
    parser.add_argument("--interval", type=float, default=1.0, help="Sampling interval seconds")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="How often to poll for process name matches",
    )
    parser.add_argument("--csv", type=Path, default=Path("pid_memory.csv"), help="CSV output path")
    parser.add_argument("--png", type=Path, default=Path("pid_memory.png"), help="PNG output path")
    return parser.parse_args()


def find_process_by_name_exact(name_exact: str):
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            info = proc.info
            proc_name = info.get("name") or ""
            if proc_name == name_exact:
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return None


def resolve_target_process(args):
    if args.pid is not None:
        try:
            return psutil.Process(args.pid)
        except psutil.NoSuchProcess:
            print(f"PID {args.pid} does not exist.")
            return None

    print(
        f"Waiting for process with exact name '{args.name}' "
        f"(poll every {args.poll_interval}s)..."
    )
    while True:
        proc = find_process_by_name_exact(args.name)
        if proc is not None:
            return proc
        time.sleep(args.poll_interval)


def generate_plot(csv_path: Path, png_path: Path, process_name: str):
    timestamps = []
    rss_mib = []
    swap_mib = []
    oom_scores = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamps.append(datetime.fromisoformat(row["timestamp_utc"]))
            rss_mib.append(float(row["rss_mib"]))
            swap_mib.append(float(row["swap_mib"]))
            oom_scores.append(int(row["oom_score"]) if row["oom_score"] else None)

    if not timestamps:
        print("No samples collected, skipping plot.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 4.5))

    # Memory on left y-axis
    ax1.plot(timestamps, rss_mib, linewidth=2, label="RSS (MiB)", color="tab:blue")
    ax1.plot(timestamps, swap_mib, linewidth=2, label="Swap (MiB)", color="tab:orange")
    ax1.set_xlabel("Time (UTC)")
    ax1.set_ylabel("Memory (MiB)")
    ax1.grid(True, alpha=0.3)

    # OOM score on right y-axis (if available)
    has_oom_data = any(s is not None for s in oom_scores)
    if has_oom_data:
        ax2 = ax1.twinx()
        ax2.plot(timestamps, oom_scores, linewidth=2, label="OOM Score", color="tab:red", linestyle="--")
        ax2.set_ylabel("OOM Score (0-1000)")
        ax2.set_ylim(0, 1000)
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    else:
        ax1.legend(loc="upper left")

    ax1.set_title(f"Process Memory Over Time ({process_name})")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {png_path}")


def get_top_processes(n: int) -> list[dict]:
    """Get top N processes by RSS memory usage."""
    procs = []
    for proc in psutil.process_iter(["pid", "name", "memory_info"]):
        try:
            info = proc.info
            mem_info = info.get("memory_info")
            if mem_info is None:
                continue
            procs.append({
                "pid": info["pid"],
                "name": info["name"] or "unknown",
                "rss": mem_info.rss,
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    # Sort by RSS descending and return top N
    procs.sort(key=lambda x: x["rss"], reverse=True)
    return procs[:n]


def generate_top_plot(csv_path: Path, png_path: Path):
    """Generate a plot showing RSS over time for multiple processes."""
    # Read CSV and organize by process key (pid:name)
    from collections import defaultdict

    process_data = defaultdict(lambda: {"timestamps": [], "rss_mib": []})
    all_timestamps = set()

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = datetime.fromisoformat(row["timestamp_utc"])
            all_timestamps.add(ts)
            # Use pid:name as key to handle pid reuse
            proc_key = f"{row['pid']}:{row['process_name']}"
            process_data[proc_key]["timestamps"].append(ts)
            process_data[proc_key]["rss_mib"].append(float(row["rss_mib"]))
            process_data[proc_key]["name"] = row["process_name"]
            process_data[proc_key]["pid"] = row["pid"]

    if not process_data:
        print("No samples collected, skipping plot.")
        return

    # Find processes with highest peak RSS for legend ordering
    peak_rss = {k: max(v["rss_mib"]) for k, v in process_data.items()}
    sorted_procs = sorted(process_data.keys(), key=lambda k: peak_rss[k], reverse=True)

    # Limit legend to top 15 processes, group rest as "others"
    max_legend = 15
    fig, ax = plt.subplots(figsize=(14, 8))

    # Use a colormap that handles many processes
    colors = plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors

    for i, proc_key in enumerate(sorted_procs[:max_legend]):
        data = process_data[proc_key]
        color = colors[i % len(colors)]
        label = f"{data['name']} (pid {data['pid']}, peak {peak_rss[proc_key]:.0f} MiB)"
        ax.plot(data["timestamps"], data["rss_mib"], linewidth=1.5, label=label, color=color)

    # Plot remaining processes with thin gray lines (no legend)
    for proc_key in sorted_procs[max_legend:]:
        data = process_data[proc_key]
        ax.plot(data["timestamps"], data["rss_mib"], linewidth=0.5, color="gray", alpha=0.3)

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("RSS (MiB)")
    ax.set_title(f"Top Memory Consumers Over Time ({len(process_data)} processes tracked)")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))

    # Place legend outside plot
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot: {png_path}")


def run_top_monitor(args):
    """Monitor top N memory-consuming processes system-wide."""
    seen_processes = set()  # Track all processes we've seen

    with args.csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_utc", "pid", "process_name", "rss_bytes", "rss_mib", "swap_bytes", "swap_mib"])

        print(
            f"Monitoring top {args.top} memory-consuming processes every {args.interval}s.\n"
            f"Writing CSV: {args.csv}\nPress Ctrl+C to stop and generate {args.png}."
        )

        try:
            while True:
                ts = datetime.now(timezone.utc).isoformat()
                top_procs = get_top_processes(args.top)

                for proc in top_procs:
                    pid = proc["pid"]
                    name = proc["name"]
                    rss_bytes = proc["rss"]
                    swap_bytes = get_swap_bytes(pid)

                    seen_processes.add(f"{pid}:{name}")

                    writer.writerow([
                        ts,
                        pid,
                        name,
                        rss_bytes,
                        round(bytes_to_mib(rss_bytes), 3),
                        swap_bytes,
                        round(bytes_to_mib(swap_bytes), 3),
                    ])

                f.flush()
                time.sleep(args.interval)

        except KeyboardInterrupt:
            print(f"\nStopping sampler... tracked {len(seen_processes)} unique processes.")

    generate_top_plot(args.csv, args.png)


def infer_process_name_from_csv(csv_path: Path) -> str:
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        first_row = next(reader, None)
        if first_row is None:
            return csv_path.stem
        process_name = (first_row.get("process_name") or "").strip()
        return process_name or csv_path.stem


def main():
    args = parse_args()

    if args.from_csv is not None:
        # Detect if this is a top-monitor CSV (no oom_score column) or single-process CSV
        with args.from_csv.open("r", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
        if "oom_score" not in fieldnames:
            generate_top_plot(args.from_csv, args.png)
        else:
            process_name = infer_process_name_from_csv(args.from_csv)
            generate_plot(args.from_csv, args.png, process_name)
        return

    if args.top is not None:
        run_top_monitor(args)
        return

    try:
        proc = resolve_target_process(args)
    except KeyboardInterrupt:
        print("\nStopped before attaching to a process.")
        return
    if proc is None:
        return
    target_pid = proc.pid
    proc_name = proc.name()
    target_create_time = proc.create_time()

    with args.csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_utc", "pid", "process_name", "rss_bytes", "rss_mib", "swap_bytes", "swap_mib", "oom_score"])

        print(
            f"Sampling RSS, swap, and OOM score for PID={target_pid} ({proc_name}) every {args.interval}s.\n"
            f"Writing CSV: {args.csv}\nPress Ctrl+C to stop and generate {args.png}."
        )

        try:
            while True:
                # Process may exit between iterations
                try:
                    if not proc.is_running() or proc.create_time() != target_create_time:
                        print(f"PID {target_pid} exited/restarted; stopping sampler.")
                        break
                    # Do not use memory_full_info() in this hot loop: on Linux it
                    # can be expensive (often parsing detailed memory maps), which
                    # distorts high-frequency sampling and perturbs the target app.
                    mem_info = proc.memory_info()
                    rss_bytes = mem_info.rss
                    swap_bytes = get_swap_bytes(target_pid)
                    oom_score = get_oom_score(target_pid)
                except psutil.NoSuchProcess:
                    print(f"PID {target_pid} exited; stopping sampler.")
                    break

                ts = datetime.now(timezone.utc).isoformat()
                writer.writerow([
                    ts,
                    target_pid,
                    proc_name,
                    rss_bytes,
                    round(bytes_to_mib(rss_bytes), 3),
                    swap_bytes,
                    round(bytes_to_mib(swap_bytes), 3),
                    oom_score if oom_score is not None else "",
                ])
                f.flush()
                time.sleep(args.interval)

        except KeyboardInterrupt:
            print("\nStopping sampler...")

    generate_plot(args.csv, args.png, proc_name)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import os
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(os.environ.get("POLY5M_ROOT", Path(__file__).resolve().parents[1])).expanduser()


def run_json(command: list[str]) -> dict:
    env = os.environ.copy()
    src_path = str((ROOT / "src").resolve())
    env["PYTHONPATH"] = src_path + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else src_path
    env["POLY5M_ROOT"] = str(ROOT)
    completed = subprocess.run(command, check=True, capture_output=True, text=True, env=env)
    return json.loads(completed.stdout)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile and report paper-trading evidence")
    parser.add_argument("--output-dir", default="outputs/paper_trader")
    parser.add_argument("--report-dir", default="outputs/paper_trader/reports")
    parser.add_argument("--since", default=None)
    parser.add_argument("--until", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    report_dir = Path(args.report_dir).expanduser()
    if not report_dir.is_absolute():
        report_dir = ROOT / report_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    reconcile = run_json(
        [
            sys.executable,
            str(ROOT / "scripts/reconcile_official_state.py"),
            "--config",
            str(ROOT / "config/paper_btc_5m.json"),
            "--output-dir",
            str(output_dir),
        ]
    )
    audit = run_json([sys.executable, str(ROOT / "scripts/audit_paper_trades.py"), "--output-dir", str(output_dir)])
    performance = run_json([sys.executable, str(ROOT / "scripts/performance_report.py"), "--output-dir", str(output_dir)])

    (report_dir / f"reconcile_{stamp}.json").write_text(
        json.dumps(reconcile, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (report_dir / f"audit_{stamp}.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (report_dir / f"performance_{stamp}.json").write_text(
        json.dumps(performance, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    evidence_command = [
        sys.executable,
        str(ROOT / "scripts/generate_evidence_report.py"),
        "--output-dir",
        str(output_dir),
        "--report-dir",
        str(report_dir),
    ]
    if args.since:
        evidence_command.extend(["--since", args.since])
    if args.until:
        evidence_command.extend(["--until", args.until])
    evidence = run_json(evidence_command)

    print(
        json.dumps(
            {
                "reconcile": str(report_dir / f"reconcile_{stamp}.json"),
                "audit": str(report_dir / f"audit_{stamp}.json"),
                "performance": str(report_dir / f"performance_{stamp}.json"),
                "evidence": evidence,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

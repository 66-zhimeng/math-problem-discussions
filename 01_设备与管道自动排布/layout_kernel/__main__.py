"""命令行入口：

    .venv/Scripts/python -m layout_kernel 任务书.json -o 方案.json [-q]

在 01_设备与管道自动排布 目录下运行（或把该目录加入 PYTHONPATH）。
"""
import argparse
import json
import sys
from pathlib import Path

from .api import solve
from .contract import CaseError


def main(argv=None):
    ap = argparse.ArgumentParser(prog="layout_kernel", description="设备与管道自动排布计算内核")
    ap.add_argument("case", help="任务书 JSON 路径")
    ap.add_argument("-o", "--out", help="结果 JSON 路径（不给则打印摘要）")
    ap.add_argument("-q", "--quiet", action="store_true", help="不输出过程日志")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        r = solve(a.case, log=None if a.quiet else (lambda line: print(line, flush=True)))
    except CaseError as ex:
        print(f"任务书不合法：{ex}", file=sys.stderr)
        return 2
    if a.out:
        Path(a.out).write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"结果已写入 {a.out}")
    m = r.get("metrics", {})
    print(f"{'成功' if r['ok'] else '未达成'}：用时 {r['total_s']} s" +
          (f"，J={m.get('J')}，占地 {m.get('area_m2')} m²，管长 {m.get('L_m')} m，"
           f"弯头 {m.get('bends')}，高度变化 {m.get('height_changes')}" if m else ""))
    for v in r.get("violations", [])[:20]:
        print(f"  违规：{v}")
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())

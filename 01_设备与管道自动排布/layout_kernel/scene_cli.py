"""三维场景任务的命令行入口：标准输入读请求 JSON，标准输出写结果 JSON（供其他项目以子进程调用，
调用方不需要安装本内核的依赖）。

请求：{"input": 场景输入（见 scene.py）, "settings": 布管设置（见 scene.SETTINGS_REQUIRED）}
结果：scene.optimize_scene 的返回值（布管 + 可移动节点的平移）；出错时 {"ok": false, "error": "…"}，退出码 1。
过程日志写到标准错误。

    python -m layout_kernel.scene_cli < 请求.json > 结果.json
"""
import json
import sys

from . import scene


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    try:
        req = json.loads(sys.stdin.buffer.read().decode("utf-8"))
        for k in ("input", "settings"):
            if k not in req:
                raise KeyError(f"请求缺少 {k}")
        out = scene.optimize_scene(req["input"], req["settings"], log=lambda line: print(line, file=sys.stderr, flush=True))
    except Exception as ex:                                        # 以 JSON 报错，调用方统一处理
        sys.stdout.write(json.dumps({"ok": False, "error": f"{type(ex).__name__}: {ex}"}, ensure_ascii=False))
        return 1
    sys.stdout.write(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

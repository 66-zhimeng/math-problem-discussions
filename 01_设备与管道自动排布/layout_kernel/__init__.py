"""设备与管道自动排布：计算内核。

外部只通过本包调用：

    from layout_kernel import solve, validate_case, CaseError
    result = solve("算例.json")          # 或传 dict

契约（输入任务书与输出方案的字段）见 layout_kernel/契约.md。
"""
from .api import solve
from .contract import CaseError, validate_case

__all__ = ["solve", "validate_case", "CaseError"]

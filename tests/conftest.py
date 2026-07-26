"""pytest 运行环境：使 lxa_aat 包与 comfy 核心可导入（无本机绝对路径）。"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]   # lxa_aat 包目录
_PARENT = _REPO.parent                        # custom_nodes/
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

# 向上寻找 ComfyUI 根（含 comfy/ 包的目录），保证 `import comfy.*` 可用
for _d in [_REPO, *_REPO.parents]:
    if (_d / "comfy").is_dir():
        if str(_d) not in sys.path:
            sys.path.insert(0, str(_d))
        break

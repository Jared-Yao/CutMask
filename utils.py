import os, random
import numpy as np
import torch

def str2bool(v):
    """argparse handels type=bool in a weird way.
    See this stack overflow: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    we can use this function as type converter for boolean values
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
# === 新增：统一设种子 ===
def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    统一设置 Python / NumPy / PyTorch 的随机种子。
    deterministic=True 时，尽量强制确定性（可能降低速度）。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        # 关闭 CuDNN 的自动调优，启用确定性内核（老办法，依然有用）
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # 新办法：要求使用确定性算法（PyTorch>=1.8）
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        # 对 CUDA≥10.2 的若干算子，要求设置 cuBLAS 工作区环境变量
        # 注意：严格来说要在进程启动前设置，这里兜底设置一份。
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# === 新增：给 DataLoader 的每个 worker 设 NumPy / random 的种子 ===
def seed_worker(worker_id: int):
    # 按官方建议，用 initial_seed 派生 32-bit 种子给 numpy/random
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
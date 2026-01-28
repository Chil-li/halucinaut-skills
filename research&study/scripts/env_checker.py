"""
Environment Diagnostic Tool for Paper Reproduction Assistant
Usage: python env_checker.py
Output: System capability JSON report
"""
import sys
import platform
import json
import os
from datetime import datetime

def get_gpu_info():
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False, "reason": "CUDA not available"}
        
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "name": props.name,
                "vram_gb": round(props.total_memory / (1024**3), 2),
                "capability": f"{props.major}.{props.minor}"
            })
        
        return {
            "available": True,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": len(gpus),
            "devices": gpus
        }
    except ImportError:
        return {"available": False, "reason": "PyTorch not installed"}
    except Exception as e:
        return {"available": False, "error": str(e)}

def main():
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": sys.version.split()[0],
        "os": platform.platform(),
        "gpu_info": get_gpu_info(),
        # 关键：检查是否存在过时的编译工具，这常导致 FlashAttn 安装失败
        "gcc_check": os.popen("gcc --version").read().split('\n')[0] if os.name != 'nt' else "Windows (Check Visual Studio)"
    }
    
    # 直接打印 JSON 供 Assistant 解析
    print("---BEGIN_ENV_REPORT---")
    print(json.dumps(report, indent=2))
    print("---END_ENV_REPORT---")

if __name__ == "__main__":
    main()
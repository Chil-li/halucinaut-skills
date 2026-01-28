"""
Standard Mock Tester Template for Paper Reproduction
此脚本用于在完整训练前验证模型的 Data Flow 和梯度 Gradient。
"""
import torch
import torch.nn as nn
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [MOCK] - %(message)s')
logger = logging.getLogger(__name__)

def run_mock_test(model_class, input_shape, device='cpu'):
    logger.info(f"Starting Mock Test for {model_class.__name__}...")
    
    # 1. Device Check
    device = torch.device(device)
    logger.info(f"Running on device: {device}")

    try:
        # 2. Model Initialization
        model = model_class().to(device)
        logger.info("Model initialized successfully.")
        
        # 3. Input Shape Verification
        # Generate random tensor matching strict input specs
        dummy_input = torch.randn(*input_shape).to(device)
        logger.info(f"Dummy input created with shape: {dummy_input.shape}")

        # 4. Forward Pass
        output = model(dummy_input)
        logger.info(f"Forward pass successful. Output shape: {output.shape}")
        
        # 5. NaN Check
        if torch.isnan(output).any():
            raise ValueError("Output contains NaN values!")
            
        # 6. Backward Pass (Gradient Check)
        # Assume a simple loss for gradient checking
        if isinstance(output, tuple):
             # Handle models that return tuples (e.g., RNNs, Transformers)
             target = output[0].mean()
        else:
             target = output.mean()
             
        target.backward()
        logger.info("Backward pass successful. Gradients computed.")

        # 7. Parameter Update Check
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                break
        
        if not has_grad:
            logger.warning("No gradients found! Check if require_grad is set correctly.")
        else:
            logger.info("Gradient flow confirmed.")

        logger.info(">>> MOCK TEST PASSED <<<")
        return True

    except Exception as e:
        logger.error(f"Mock Test FAILED: {str(e)}")
        # 打印部分堆栈以便调试 (在实际 Skill 中由 LLM 分析)
        import traceback
        traceback.print_exc()
        return False

# --- LLM INSTRUCTION ---
# 当你需要生成 Mock Test 时，引用此文件。
# 仅需替换 model_class 和 input_shape。
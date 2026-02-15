@echo off
chcp 65001 > nul
echo ========================================================
echo [Genius Protocol] RWKV Piano Muse O(T) 并行预填充推理引擎
echo Target Env: C:\Users\nicho\anaconda3\python.exe
echo ========================================================

uv run --python C:\Users\nicho\anaconda3\python.exe infer_copilot.py ^
    --model_path ./models/best_model.pth ^
    --context_midi ./examples/context.mid ^
    --output_dir ./outputs ^
    --max_new_tokens 512 ^
    --temperature 0.85 ^
    --top_p 0.90

pause

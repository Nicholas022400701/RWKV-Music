@echo off
chcp 65001 > nul
echo ========================================================
echo [Genius Protocol] RWKV Piano Muse 物理级并行训练启动
echo Target Env: C:\Users\nicho\anaconda3\python.exe
echo ========================================================

:: 强行绑定并在当前环境中用 uv 起步，避免隔离问题
uv run --python C:\Users\nicho\anaconda3\python.exe train_parallel.py ^
    --data_path ./data/processed/processed_dataset.jsonl ^
    --pretrained_model ./models/rwkv_base.pth ^
    --output_dir ./models ^
    --batch_size 4 ^
    --max_seq_len 1024 ^
    --epochs 10

pause

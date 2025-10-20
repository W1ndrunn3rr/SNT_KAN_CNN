monitor-gpu:
    watch -n 1 -c "nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv"

tensorboard:
    tensorboard --logdir logs/tensorboard --port 6006 --bind_all

clear-logs:
    cd logs && rm -rf * && cd ..

train MODEL:
    uv run train model_params.model_type={{MODEL}}

test CHECKPOINT:
    uv run test test_params.checkpoint_path={{CHECKPOINT}}

test-last MODEL:
    uv run test test_params.checkpoint_path=logs/models/{{MODEL}}-last.ckpt

clone-kan-repo:
    git clone https://github.com/ZiyaoLi/fast-kan

get-sar-data:
    uv run python src/scripts/get_sar_data.py

prepare-env:
    just clone-kan-repo && just get-sar-data

prepare-data: 
    uv run python src/scripts/prepare_data.py

analyze-data:
    uv run python src/scripts/analyze_data.py
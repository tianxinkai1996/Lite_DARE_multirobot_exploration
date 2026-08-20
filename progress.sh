#!/usr/bin/env bash
cd "/root/lite dare/DARE" || exit 1

PY="/root/miniconda3/envs/env_dare/bin/python"

TARGET=$("$PY" -c 'from dataset_parameter import NUM_EPISODES; print(NUM_EPISODES)' 2>/dev/null || echo "?")

DONE=$("$PY" -c '
import zarr
from dataset_parameter import dataset_path
z = zarr.open_group(f"{dataset_path}/data.zarr", mode="r")
print(len(z["meta"]["episode_ends"]))
' 2>/dev/null || echo "?")

LAST=$(grep -oE 'starting episode [0-9]+' logs/dataset_driver.log 2>/dev/null \
  | tail -1 | awk '{print $3}')

WORKERS=$(pgrep -fc 'ray::Runner.job' 2>/dev/null || true)
DRIVER=$(pgrep -fc 'dataset_driver.py' 2>/dev/null || true)

echo "========== DARE 实时进度 =========="
echo "主程序数量: ${DRIVER}"
echo "活跃 Ray Runner: ${WORKERS}"
echo "已写入 Zarr episodes（含之前已有数据）: ${DONE}"

if [[ "$LAST" =~ ^[0-9]+$ && "$TARGET" =~ ^[0-9]+$ ]]; then
    ASSIGNED=$((LAST + 1))
    PCT=$(awk -v a="$ASSIGNED" -v t="$TARGET" 'BEGIN {printf "%.2f", a*100/t}')
    echo "本轮已分配到: ${ASSIGNED} / ${TARGET} (${PCT}%)"
else
    echo "本轮已分配到: 暂未从日志读取到 episode 编号"
fi

echo
echo "========== 最新日志 =========="
tail -n 8 logs/dataset_driver.log

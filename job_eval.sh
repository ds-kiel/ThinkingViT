#!/bin/bash -l
#SBATCH --job-name=ThinViT-eval
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128000
#SBATCH --time=1-00:00:00
#SBATCH --output=eval.out
#SBATCH --error=eval.err
#SBATCH --partition=long
#SBATCH --gres=gpu:L40:1

set -euo pipefail

if ! command -v module &> /dev/null; then
    set +u
    source /etc/profile || true
    set -u
fi

for MODULE_INIT in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash; do
    if ! command -v module &> /dev/null && [ -f "${MODULE_INIT}" ]; then
        set +u
        source "${MODULE_INIT}" || true
        set -u
    fi
done

if command -v module &> /dev/null; then
    module load gpu-env
    module load cuda
    module load gcc12-env
    module load python
else
    echo "Warning: environment module command not found; continuing with the current environment."
fi

source /home/aho/envs/env/bin/activate

cd /home/aho/ThinkingViTCVPR/ThinkingViT

CHECKPOINT="/home/aho/ThinkingViTCVPR/ThinkingViT/ThinkingViT_re_3_6.pth.tar"
DATA_DIR="/data22/datasets/ilsvrc2012/"
BATCH_SIZE=512
MODEL="thinkingvit"
THINKING_STAGES=(3 6)
EMA_FLAG="--use-ema"

# Edit this list to choose the thresholds to evaluate.
THRESHOLDS=(0.0 0.1 0.2 0.3 0.5 0.8 1.0 1.2 1.4 1.6 2 5)

LOG_DIR="eval_logs"
mkdir -p "${LOG_DIR}"
rm -f "${LOG_DIR}"/threshold_*.log "${LOG_DIR}/summary.md"

for THRESHOLD in "${THRESHOLDS[@]}"; do
    SAFE_THRESHOLD="${THRESHOLD//./p}"
    echo "Evaluating threshold ${THRESHOLD}"

    srun python validate.py \
        --model "${MODEL}" \
        --checkpoint "${CHECKPOINT}" \
        --data-dir "${DATA_DIR}" \
        --batch-size "${BATCH_SIZE}" \
        ${EMA_FLAG} \
        --thinking_stages "${THINKING_STAGES[@]}" \
        --threshold "${THRESHOLD}" \
        &> "${LOG_DIR}/threshold_${SAFE_THRESHOLD}.log"
done

python - "${LOG_DIR}" "${CHECKPOINT}" "${MODEL}" "${THINKING_STAGES[*]}" <<'PY'
import json
import re
import sys
from datetime import datetime
from pathlib import Path

log_dir = Path(sys.argv[1])
checkpoint = sys.argv[2]
model = sys.argv[3]
thinking_stages = sys.argv[4]

rows = []
max_stages = 0

for log_path in sorted(log_dir.glob("threshold_*.log")):
    text = log_path.read_text(errors="replace")
    threshold_match = re.search(r"Entropy Threshold:\s*([0-9.]+)", text)
    top1_match = re.search(r"\*\s+Acc@1\s+([0-9.]+)", text)
    top5_match = re.search(r"Acc@5\s+([0-9.]+)", text)
    gmacs_match = re.search(r"GMACs:\s*([0-9.]+)", text)
    result_match = re.search(r"--result\s*(\{.*?\})\s*$", text, re.DOTALL)

    top1 = top1_match.group(1) if top1_match else ""
    top5 = ""
    if result_match:
        try:
            result = json.loads(result_match.group(1))
            top1 = str(result.get("top1", top1))
            top5 = str(result.get("top5", ""))
        except json.JSONDecodeError:
            pass
    elif top5_match:
        top5 = top5_match.group(1)

    dispatches = re.findall(
        r"Model\s+\d+\s+Dispatched:\s*([0-9.]+)%\s+\((\d+)/(\d+)\)",
        text,
    )
    stage_accs = re.findall(
        r"Model\s+\d+\s+Accuracy:\s*([0-9.]+)%\s+\((\d+)/(\d+)\)",
        text,
    )
    max_stages = max(max_stages, len(dispatches), len(stage_accs))

    threshold = threshold_match.group(1) if threshold_match else log_path.stem.replace("threshold_", "").replace("p", ".")
    rows.append({
        "threshold": threshold,
        "top1": top1,
        "top5": top5,
        "gmacs": gmacs_match.group(1) if gmacs_match else "",
        "dispatches": dispatches,
        "stage_accs": stage_accs,
        "log": log_path.name,
    })

rows.sort(key=lambda row: float(row["threshold"]))

headers = ["Threshold", "Top-1", "Top-5", "GMACs"]
for idx in range(max_stages):
    headers.append(f"Stage {idx + 1} Dispatch")
for idx in range(max_stages):
    headers.append(f"Stage {idx + 1} Acc")
headers.append("Log")

lines = [
    "# ThinkingViT Evaluation Summary",
    "",
    f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    f"- Model: `{model}`",
    f"- Thinking stages: `{thinking_stages}`",
    f"- Checkpoint: `{checkpoint}`",
    "",
    "|" + "|".join(headers) + "|",
    "|" + "|".join(["---"] * len(headers)) + "|",
]

for row in rows:
    cells = [
        row["threshold"],
        row["top1"],
        row["top5"],
        row["gmacs"],
    ]
    for idx in range(max_stages):
        if idx < len(row["dispatches"]):
            pct, count, total = row["dispatches"][idx]
            cells.append(f"{pct}% ({count}/{total})")
        else:
            cells.append("")
    for idx in range(max_stages):
        if idx < len(row["stage_accs"]):
            pct, correct, total = row["stage_accs"][idx]
            cells.append(f"{pct}% ({correct}/{total})")
        else:
            cells.append("")
    cells.append(f"[{row['log']}](./{row['log']})")
    lines.append("|" + "|".join(cells) + "|")

report_path = log_dir / "summary.md"
report_path.write_text("\n".join(lines) + "\n")
print(f"Wrote evaluation summary to {report_path}")
PY

if command -v jobinfo &> /dev/null; then
    jobinfo
fi

#!/usr/bin/env bash
set -euo pipefail

# --- resolve repo root (script is in repo_root/scripts) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"


export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"



# --- GPU pinning ---
export CUDA_VISIBLE_DEVICES=3

# --- user knobs ---
BASE_CFG="${REPO_ROOT}/config/main.yaml"
OUT_ROOT="${REPO_ROOT}/runs"

# Edit these lists to your actual options:
DATASETS=(boiler pam mitecg state)
CLFS=(gru)
XAI=(winit jimex binwinit ig deeplift gradientshap biwinit )   # example names
CV_FOLDS=(0)

# run command
RUN_CMD="python -u -m src.run --config"
# RUN_CMD="python -u ${REPO_ROOT}/src/run.py --config"
# RUN_CMD="python -u -m src.run --config"

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

# --- timing tool detection (GNU time preferred) ---
TIME_CMD=""
for c in /usr/bin/time /bin/time /usr/local/bin/gtime; do
  [[ -x "$c" ]] && TIME_CMD="$c" && break
done


for data in "${DATASETS[@]}"; do
  for clf in "${CLFS[@]}"; do
    for expl in "${XAI[@]}"; do
      for fold in "${CV_FOLDS[@]}"; do

        TAG="${data}__${clf}__${expl}__cv${fold}"
        CFG_DIR="${REPO_ROOT}/config/_auto"; mkdir -p "$CFG_DIR"
        CFG_PATH="${CFG_DIR}/${TAG}.yaml"

        RUN_DIR="${OUT_ROOT}/${TAG}"
        OUT_DIR="${RUN_DIR}/output"
        CKPT_DIR="${RUN_DIR}/ckpt"
        PLOT_DIR="${RUN_DIR}/plots"
        LOG_DIR="${RUN_DIR}/logs"
        mkdir -p "$OUT_DIR" "$CKPT_DIR" "$PLOT_DIR" "$LOG_DIR"
        LOG_FILE="${LOG_DIR}/run.log"

        echo "[$(timestamp)] ▶ ${TAG} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"

        # --- write temp YAML by tweaking the base config ---
        python - <<PY
import yaml, os, pathlib
base = "${BASE_CFG}"
out  = "${CFG_PATH}"
with open(base, "r") as f:
    cfg = yaml.safe_load(f)

cfg["data"] = "${data}"

key = "classifier"
for cand in ("classifier","model","clf"):
    if cand in cfg:
        key = cand; break
cfg[key] = "${clf}"

cfg["explainer"] = ["${expl}"]
cfg["cv"] = [int(${fold})]

cfg["outpath"]  = "${OUT_DIR}/"
cfg["ckptpath"] = "${CKPT_DIR}/"
cfg["plotpath"] = "${PLOT_DIR}/"
cfg["logpath"]  = "${LOG_DIR}/"
cfg["resultfile"] = "results.csv"
cfg["logfile"] = "run.log"

pathlib.Path(os.path.dirname(out)).mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

        # skip if already complete
        if [[ -s "${RUN_DIR}/results.csv" ]]; then
          echo "[$(timestamp)] ⏭  results.csv exists; skipping ${TAG}"
          continue
        fi

        # run & tee logs
        # ------ run & tee logs ------
        if [[ -n "$TIME_CMD" ]]; then
        ( cd "${REPO_ROOT}" && \
            "$TIME_CMD" -f "wall=%E cpu=%P mem=%MKB" \
            ${RUN_CMD} "${CFG_PATH}" 2>&1 | tee "${LOG_FILE}" )
        else
        echo "[warn] GNU time not found; running without resource timings."
        ( cd "${REPO_ROOT}" && \
            ${RUN_CMD} "${CFG_PATH}" 2>&1 | tee "${LOG_FILE}" )
        fi


        echo "[$(timestamp)] ✅ done: ${TAG}"
        echo
      done
    done
  done
done

echo "[$(timestamp)] 🎉 sweep complete. Outputs in ${OUT_ROOT}/"

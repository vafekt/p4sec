#!/bin/bash
# run_all_combos.sh
#
# Runs every combination of:
#   reduction  : pca k={3,5,8,10} bits={16,32}
#   model      : dt | rf
# for the full pipeline steps 2 → 3 → 4 → 5.
#
# Results per combo are saved under:
#   results/<combo_name>/
#     tables/          (all files from control_plane/tables/)
#     model/           (trained model file from step 3)
#     basic.p4
#     p4sec_tofino.p4
#     pipeline.log     (stdout/stderr of this combo's run)
#
# A master log is written to results/run_all_combos.log

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CP_DIR="$ROOT_DIR/control_plane"
TABLES="$CP_DIR/tables"
MODEL="$CP_DIR/model"
RESULTS_DIR="$ROOT_DIR/results"
MASTER_LOG="$RESULTS_DIR/run_all_combos.log"

# ── Logging ───────────────────────────────────────────────────────────────────
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo "$msg"
    echo "$msg" >> "$MASTER_LOG"
}

# ── Single-combo runner ───────────────────────────────────────────────────────
# run_combo <components> <bits> <model>
run_combo() {
    local components="$1"
    local bits="$2"
    local model="$3"

    local folder_name="pca_k${components}_b${bits}_${model}"
    local result_dir="$RESULTS_DIR/$folder_name"
    local combo_log="$result_dir/pipeline.log"
    local t_start
    t_start=$(date +%s)

    log "========== START: $folder_name =========="
    mkdir -p "$result_dir"
    : > "$combo_log"

    # ── Reset shared working dirs ──────────────────────────────────────────
    rm -rf "$TABLES" && mkdir -p "$TABLES"
    mkdir -p "$MODEL"

    # ── Step 2: PCA dimensionality reduction ──────────────────────────────
    log "  [2] PCA — k=$components components, $bits-bit quantisation"
    python "$CP_DIR/2_pca_generate_entries.py" \
        --components "$components" \
        --bits "$bits" \
        >> "$combo_log" 2>&1
    if [ $? -ne 0 ]; then
        log "  FAILED at step 2 — see $result_dir/pipeline.log"; return 1
    fi

    # ── Step 3: Train model ────────────────────────────────────────────────
    log "  [3] Training model: $model"
    python "$CP_DIR/3_train_model.py" \
        --model-type "$model" \
        -i "$TABLES/transform_mapping.csv" \
        -o "$MODEL/${model}.model" \
        >> "$combo_log" 2>&1
    if [ $? -ne 0 ]; then
        log "  FAILED at step 3 — see $result_dir/pipeline.log"; return 1
    fi

    # ── Step 4: Generate model P4 table entries ────────────────────────────
    log "  [4] Generating model entries"
    local tree_suffix="tree"
    [ "$model" = "rf" ] && tree_suffix="trees"
    python "$CP_DIR/4_generate_model_entries.py" \
        --model-type "$model" \
        -i "$MODEL/${model}.model" \
        -o "$TABLES/s1-commands.txt" \
        --csv "$TABLES/transform_mapping.csv" \
        --tree-out "$TABLES/${model}_${tree_suffix}.txt" \
        --params "$TABLES/${model}_params.json" \
        >> "$combo_log" 2>&1
    if [ $? -ne 0 ]; then
        log "  FAILED at step 4 — see $result_dir/pipeline.log"; return 1
    fi

    # ── Step 5: Generate P4 programs ──────────────────────────────────────
    log "  [5] Generating P4 code"
    python "$CP_DIR/5_generating_p4_code.py" \
        --model-type "$model" \
        --output "$ROOT_DIR/basic.p4" \
        --tofino-output "$ROOT_DIR/p4sec_tofino.p4" \
        --params-file "$TABLES/encoding_params.json" \
        --commands-file "$TABLES/s1-commands.txt" \
        --reduction-config "$TABLES/reduction_config.json" \
        --rf-params "$TABLES/rf_params.json" \
        --xgb-params "$TABLES/xgb_params.json" \
        >> "$combo_log" 2>&1
    if [ $? -ne 0 ]; then
        log "  FAILED at step 5 — see $result_dir/pipeline.log"; return 1
    fi

    # ── Collect results ────────────────────────────────────────────────────
    mkdir -p "$result_dir/tables"
    cp -r "$TABLES/." "$result_dir/tables/"

    mkdir -p "$result_dir/model"
    [ -f "$MODEL/${model}.model" ] && cp "$MODEL/${model}.model" "$result_dir/model/"

    for f in basic.p4 p4sec_tofino.p4; do
        [ -f "$ROOT_DIR/$f" ] && cp "$ROOT_DIR/$f" "$result_dir/"
    done

    local elapsed=$(( $(date +%s) - t_start ))
    log "  Saved  -> results/$folder_name/  (${elapsed}s)"
    log "========== DONE:  $folder_name =========="
    return 0
}

# ── Combo matrix ──────────────────────────────────────────────────────────────
MODELS=("dt" "rf")
PCA_COMPONENTS=(3 5 8 10)
PCA_BITS=(16 32)

# ── Setup ─────────────────────────────────────────────────────────────────────
mkdir -p "$RESULTS_DIR"
: > "$MASTER_LOG"

TOTAL_START=$(date +%s)
PASS=()
FAIL=()

log "=== p4sec combo runner ==="
log "Root:    $ROOT_DIR"
log "Results: $RESULTS_DIR"
log ""

# ── PCA combos ────────────────────────────────────────────────────────────────
for components in "${PCA_COMPONENTS[@]}"; do
    for bits in "${PCA_BITS[@]}"; do
        for model in "${MODELS[@]}"; do
            if run_combo "$components" "$bits" "$model"; then
                PASS+=("pca_k${components}_b${bits}_${model}")
            else
                FAIL+=("pca_k${components}_b${bits}_${model}")
            fi
            echo ""
        done
    done
done

# ── Summary ───────────────────────────────────────────────────────────────────
TOTAL_ELAPSED=$(( $(date +%s) - TOTAL_START ))

log "================================================================"
log "SUMMARY  —  total time: ${TOTAL_ELAPSED}s"
log "Passed (${#PASS[@]}/$(( ${#PASS[@]} + ${#FAIL[@]} ))):"
for name in "${PASS[@]}"; do log "  [OK]  $name"; done
if [ ${#FAIL[@]} -gt 0 ]; then
    log "Failed (${#FAIL[@]}):"
    for name in "${FAIL[@]}"; do log "  [ERR] $name"; done
fi
log "================================================================"
log "Master log : $MASTER_LOG"
log "Results in : $RESULTS_DIR"

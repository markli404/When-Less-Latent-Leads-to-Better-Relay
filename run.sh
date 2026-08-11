#!/bin/bash

# ================= Configuration =================
MODEL_NAME="Qwen/Qwen3-8B"
LATENT_STEPS=40

# 1. Dataset list "gsm8k" "gpqa" "arc_easy" "arc_challenge" "mbppplus" "humanevalplus" "medqa" "aime2024" "aime2024"
DATASETS=("gsm8k" "gpqa" "humanevalplus" "medqa")

# 2. Compressor list. lobf_fast / hobf_fast are the deployable OBF path;
#    lobf / hobf are the exact-SVD reference. See README for the full set.
COMPRESSORS=("full" "gonly" "layerwise" "headwise" "lobf_fast" "hobf_fast")

# Environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ================= Main loop =================
for TASK in "${DATASETS[@]}"; do
    # --- A. Set parameters dynamically based on the task ---
    case ${TASK} in
        "gsm8k" | "arc_easy" | "arc_challenge")
            TOKENS=2048
            BS=4
            ;;
        "mbppplus" | "medqa" | "humanevalplus")
            TOKENS=4096
            BS=4
            ;;
        "gpqa")
            TOKENS=8192
            BS=2
            ;;
        "aime2024" | "aime2025")
            TOKENS=20000
            BS=2
            ;;
        *)
            TOKENS=2048
            BS=2
            ;;
    esac

    echo "########################################################"
    echo "Dataset: ${TASK} | BS: ${BS} | MaxTokens: ${TOKENS}"
    echo "########################################################"

    # --- B. Iterate through compressor methods ---
    for COMP_METHOD in "${COMPRESSORS[@]}"; do

        echo "------------------------------------------------------"
        echo "Running: Task=[${TASK}] | Compressor=[${COMP_METHOD}]"
        echo "------------------------------------------------------"

        # Run without --use_mask by default
        python -u run.py \
            --use_wandb \
            --method latent_mas \
            --model_name "${MODEL_NAME}" \
            --task "${TASK}" \
            --latent_steps ${LATENT_STEPS} \
            --prompt sequential \
            --max_samples -1 \
            --compressor "${COMP_METHOD}" \
            --max_new_tokens "${TOKENS}" \
            --generate_bs "${BS}" \
            --seed 888

    done # End Compressor Loop

    echo ""
    echo "✅ Completed Dataset: ${TASK}"
    echo ""

done # End Dataset Loop

echo "🎉 All experiments processed successfully!"

#!/bin/bash

## CPI prediction evaluation script
## Perform CPI prediction evaluation for all .bb files in ../batch_exp/simpoint_10B_1M/

BASE_DIR="../batch_exp/simpoint_10B_1M"
MODEL_PATH="./cpi_finetune_checkpoints/best_cpi_model.pt"

## Define all benchmark program names
BENCHMARKS=(
    "600.perlbench_s"
    "602.gcc_s"
    "605.mcf_s"
    "620.omnetpp_s"
    "623.xalancbmk_s"
    "625.x264_s"
    "631.deepsjeng_s"
    "641.leela_s"
    "648.exchange2_s"
    "657.xz_s"
)

echo "Starting CPI prediction evaluation..."
echo "Model path: $MODEL_PATH"
echo "Data directory: $BASE_DIR"
echo "="*50

## Perform evaluation for each benchmark program
for benchmark in "${BENCHMARKS[@]}"; do
    bb_file="${BASE_DIR}/${benchmark}.bb"
    json_file="${BASE_DIR}/${benchmark}_timeline_full_o3_o3.json"
    
    echo "Processing: $benchmark"
    echo "BB file: $bb_file"
    echo "JSON file: $json_file"
    
    ## Check if files exist
    if [[ ! -f "$bb_file" ]]; then
        echo "Error: BB file does not exist - $bb_file"
        continue
    fi
    
    if [[ ! -f "$json_file" ]]; then
        echo "Error: JSON file does not exist - $json_file"
        continue
    fi
    
    ## Execute CPI prediction evaluation
    echo "Executing command: python3 cpi_prediction_evaluation.py -i $bb_file --ref-json $json_file --model_path $MODEL_PATH"
    python3 cpi_prediction_evaluation.py -i "$bb_file" --ref-json "$json_file" --model_path "$MODEL_PATH"
    
    if [[ $? -eq 0 ]]; then
        echo "✓ $benchmark evaluation completed"
    else
        echo "✗ $benchmark evaluation failed"
    fi
    
    echo "-"*30
done

echo "All CPI prediction evaluations completed!"
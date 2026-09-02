#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=/workspaces/tagging_causality
export CUDA_VISIBLE_DEVICES="3,4"
export HDF5_USE_FILE_LOCKING=FALSE

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

# undirect, remove, inverse
llms=("openai--gpt-5.2" "anthropic--claude-opus-4.6" "google--gemini-3-pro-preview" "meta-llama--llama-3.3-70b-instruct" "qwen--qwen3.5-397b-a17b" "z-ai--glm-5" "minimax--minimax-m2.5")
types=("undirect" "remove" "inverse")
param_nrs=(1 2 3 4 5)


declare -A identifier_times
start_time=$(date +%s)

for llm in "${llms[@]}"; do
    for type in "${types[@]}"; do
        for param_nr in "${param_nrs[@]}"; do
            python ablation_direction.py --param "$param_nr" --type "$type" --llm "$llm" &
        done
    done
done
wait

# tags
llms=("openai--gpt-5.2" "anthropic--claude-opus-4.6" "google--gemini-3-pro-preview" "meta-llama--llama-3.3-70b-instruct" "qwen--qwen3.5-397b-a17b" "z-ai--glm-5" "minimax--minimax-m2.5")
types=("tags")
param_nrs=(1)
seeds=(0 1 2 3 4 5 6 7 8 9)
error_rates=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

for param_nr in "${param_nrs[@]}"; do
    for seed in "${seeds[@]}"; do
        for error_rate in "${error_rates[@]}"; do
            for llm in "${llms[@]}"; do
                for type in "${types[@]}"; do
                    python ablation_direction.py --param "$param_nr" --type "$type" --llm "$llm" --seed "$seed" --error_rate "$error_rate" &
                done
            done
            wait
        done
    done
done

end_time=$(date +%s)
total_time=$((end_time - start_time))

echo "Total computation time: $total_time seconds"
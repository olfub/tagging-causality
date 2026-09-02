#!/bin/bash

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=/workspaces/tagging_causality
export CUDA_VISIBLE_DEVICES="3,4"
export HDF5_USE_FILE_LOCKING=FALSE

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

identifiers=("bnlearn_child" "bnlearn_earthquake" "bnlearn_insurance" "bnlearn_survey" "bnlearn_asia" "bnlearn_cancer" "bnlearn_alarm" "lucas" "bnlearn_hepar2" "bnlearn_win95pts" "bnlearn_hailfinder")
llms=("anthropic--claude-opus-5" "anthropic--claude-opus-4.8" "anthropic--claude-opus-4.7" "anthropic--claude-opus-4.6")

experimental_series="E6_llm_model_versions"
order_data="random"
nr_samples=10000
seeds=(0 1 2 3 4 5 6 7 8 9)
min_samples=(2)
compute_min_prob_threshold="True"
remove_singular_tags=("False")
prior_on_weight=("False")
always_meeks=("False")
redirect_existing_edges=("True")
redirecting_strategy=(0)
include_current_edge_as_evidence=("True")
include_redirected_edges_in_edge_count=("True")
remove_duplicates="True"
anti_tags="False"

declare -A identifier_times
start_time=$(date +%s)
for min_sample in "${min_samples[@]}"; do
    for remove_singular in "${remove_singular_tags[@]}"; do
        for llm in "${llms[@]}"; do
            for prior in "${prior_on_weight[@]}"; do
                for always_meek in "${always_meeks[@]}"; do
                    for redirect in "${redirect_existing_edges[@]}"; do
                        if [ "$redirect" == "True" ]; then
                            for strategy in "${redirecting_strategy[@]}"; do
                                for include_current in "${include_current_edge_as_evidence[@]}"; do
                                    for include_redirected in "${include_redirected_edges_in_edge_count[@]}"; do
                                        for identifier in "${identifiers[@]}"; do
                                            for seed in "${seeds[@]}"; do
                                                python run_tag_only.py \
                                                    --experimental_series "$experimental_series" \
                                                    --identifier "$identifier" \
                                                    --order_data "$order_data" \
                                                    --seed "$seed" \
                                                    --load_with_llm "$llm" \
                                                    --nr_samples "$nr_samples" \
                                                    --pc_indep_test "chisq" \
                                                    --pc_alpha 0.05 \
                                                    --min_samples "$min_sample" \
                                                    --compute_min_prob_threshold "$compute_min_prob_threshold" \
                                                    --anti_tags "$anti_tags" \
                                                    --remove_duplicates "$remove_duplicates" \
                                                    --remove_singular_tags "$remove_singular" \
                                                    --prior_on_weight "$prior" \
                                                    --always_meek "$always_meek" \
                                                    --redirect_existing_edges "$redirect" \
                                                    --redirecting_strategy "$strategy" \
                                                    --min_prob_redirecting 0.6 \
                                                    --include_current_edge_as_evidence "$include_current" \
                                                    --include_redirected_edges_in_edge_count "$include_redirected" &
                                            done
                                        done
                                        wait
                                    done
                                done
                            done
                        else
                            for identifier in "${identifiers[@]}"; do
                                # redirecting parameters don't matter here (use default values)
                                for seed in "${seeds[@]}"; do
                                    python run_tag_only.py \
                                        --experimental_series "$experimental_series" \
                                        --identifier "$identifier" \
                                        --order_data "$order_data" \
                                        --seed "$seed" \
                                        --load_with_llm "$llm" \
                                        --nr_samples "$nr_samples" \
                                        --pc_indep_test "chisq" \
                                        --pc_alpha 0.05 \
                                        --min_samples "$min_sample" \
                                        --compute_min_prob_threshold "$compute_min_prob_threshold" \
                                        --anti_tags "$anti_tags" \
                                        --remove_duplicates "$remove_duplicates" \
                                        --remove_singular_tags "$remove_singular" \
                                        --prior_on_weight "$prior" \
                                        --always_meek "$always_meek" \
                                        --redirect_existing_edges "$redirect" &
                                done
                            done
                            wait
                        fi
                    done
                done
            done
        done
    done
done

identifiers=("bnlearn_child" "bnlearn_earthquake" "bnlearn_insurance" "bnlearn_survey" "bnlearn_asia" "bnlearn_cancer" "bnlearn_alarm" "lucas" "bnlearn_hepar2" "bnlearn_win95pts" "bnlearn_hailfinder")
llms=("openai--gpt-5.6-sol" "openai--gpt-5.5" "openai--gpt-5.4" "openai--gpt-5.2")

experimental_series="E6_llm_model_versions"
order_data="random"
nr_samples=10000
seeds=(0 1 2 3 4 5 6 7 8 9)
min_samples=(1)
compute_min_prob_threshold="True"
remove_singular_tags=("False")
prior_on_weight=("True")
always_meeks=("True")
redirect_existing_edges=("True")
redirecting_strategy=(1)
include_current_edge_as_evidence=("True")
include_redirected_edges_in_edge_count=("True")
remove_duplicates="True"
anti_tags="False"

for min_sample in "${min_samples[@]}"; do
    for remove_singular in "${remove_singular_tags[@]}"; do
        for llm in "${llms[@]}"; do
            for prior in "${prior_on_weight[@]}"; do
                for always_meek in "${always_meeks[@]}"; do
                    for redirect in "${redirect_existing_edges[@]}"; do
                        if [ "$redirect" == "True" ]; then
                            for strategy in "${redirecting_strategy[@]}"; do
                                for include_current in "${include_current_edge_as_evidence[@]}"; do
                                    for include_redirected in "${include_redirected_edges_in_edge_count[@]}"; do
                                        for identifier in "${identifiers[@]}"; do
                                            for seed in "${seeds[@]}"; do
                                                python run_tag_only.py \
                                                    --experimental_series "$experimental_series" \
                                                    --identifier "$identifier" \
                                                    --order_data "$order_data" \
                                                    --seed "$seed" \
                                                    --load_with_llm "$llm" \
                                                    --nr_samples "$nr_samples" \
                                                    --pc_indep_test "chisq" \
                                                    --pc_alpha 0.05 \
                                                    --min_samples "$min_sample" \
                                                    --compute_min_prob_threshold "$compute_min_prob_threshold" \
                                                    --anti_tags "$anti_tags" \
                                                    --remove_duplicates "$remove_duplicates" \
                                                    --remove_singular_tags "$remove_singular" \
                                                    --prior_on_weight "$prior" \
                                                    --always_meek "$always_meek" \
                                                    --redirect_existing_edges "$redirect" \
                                                    --redirecting_strategy "$strategy" \
                                                    --min_prob_redirecting 0.6 \
                                                    --include_current_edge_as_evidence "$include_current" \
                                                    --include_redirected_edges_in_edge_count "$include_redirected" &
                                            done
                                        done
                                        wait
                                    done
                                done
                            done
                        else
                            for identifier in "${identifiers[@]}"; do
                                # redirecting parameters don't matter here (use default values)
                                for seed in "${seeds[@]}"; do
                                    python run_tag_only.py \
                                        --experimental_series "$experimental_series" \
                                        --identifier "$identifier" \
                                        --order_data "$order_data" \
                                        --seed "$seed" \
                                        --load_with_llm "$llm" \
                                        --nr_samples "$nr_samples" \
                                        --pc_indep_test "chisq" \
                                        --pc_alpha 0.05 \
                                        --min_samples "$min_sample" \
                                        --compute_min_prob_threshold "$compute_min_prob_threshold" \
                                        --anti_tags "$anti_tags" \
                                        --remove_duplicates "$remove_duplicates" \
                                        --remove_singular_tags "$remove_singular" \
                                        --prior_on_weight "$prior" \
                                        --always_meek "$always_meek" \
                                        --redirect_existing_edges "$redirect" &
                                done
                            done
                            wait
                        fi
                    done
                done
            done
        done
    done
done

identifiers=("bnlearn_child" "bnlearn_earthquake" "bnlearn_insurance" "bnlearn_survey" "bnlearn_asia" "bnlearn_cancer" "bnlearn_alarm" "lucas" "bnlearn_hepar2" "bnlearn_win95pts" "bnlearn_hailfinder")
llms=("qwen--qwen3.8-max" "qwen--qwen3.7-max" "qwen--qwen3.6-max-preview" "qwen--qwen3.5-397b-a17b")

experimental_series="E6_llm_model_versions"
order_data="random"
nr_samples=10000
seeds=(0 1 2 3 4 5 6 7 8 9)
min_samples=(1)
compute_min_prob_threshold="True"
remove_singular_tags=("False")
prior_on_weight=("True")
always_meeks=("True")
redirect_existing_edges=("True")
redirecting_strategy=(1)
include_current_edge_as_evidence=("True")
include_redirected_edges_in_edge_count=("True")
remove_duplicates="True"
anti_tags="False"

for min_sample in "${min_samples[@]}"; do
    for remove_singular in "${remove_singular_tags[@]}"; do
        for llm in "${llms[@]}"; do
            for prior in "${prior_on_weight[@]}"; do
                for always_meek in "${always_meeks[@]}"; do
                    for redirect in "${redirect_existing_edges[@]}"; do
                        if [ "$redirect" == "True" ]; then
                            for strategy in "${redirecting_strategy[@]}"; do
                                for include_current in "${include_current_edge_as_evidence[@]}"; do
                                    for include_redirected in "${include_redirected_edges_in_edge_count[@]}"; do
                                        for identifier in "${identifiers[@]}"; do
                                            for seed in "${seeds[@]}"; do
                                                python run_tag_only.py \
                                                    --experimental_series "$experimental_series" \
                                                    --identifier "$identifier" \
                                                    --order_data "$order_data" \
                                                    --seed "$seed" \
                                                    --load_with_llm "$llm" \
                                                    --nr_samples "$nr_samples" \
                                                    --pc_indep_test "chisq" \
                                                    --pc_alpha 0.05 \
                                                    --min_samples "$min_sample" \
                                                    --compute_min_prob_threshold "$compute_min_prob_threshold" \
                                                    --anti_tags "$anti_tags" \
                                                    --remove_duplicates "$remove_duplicates" \
                                                    --remove_singular_tags "$remove_singular" \
                                                    --prior_on_weight "$prior" \
                                                    --always_meek "$always_meek" \
                                                    --redirect_existing_edges "$redirect" \
                                                    --redirecting_strategy "$strategy" \
                                                    --min_prob_redirecting 0.6 \
                                                    --include_current_edge_as_evidence "$include_current" \
                                                    --include_redirected_edges_in_edge_count "$include_redirected" &
                                            done
                                        done
                                        wait
                                    done
                                done
                            done
                        else
                            for identifier in "${identifiers[@]}"; do
                                # redirecting parameters don't matter here (use default values)
                                for seed in "${seeds[@]}"; do
                                    python run_tag_only.py \
                                        --experimental_series "$experimental_series" \
                                        --identifier "$identifier" \
                                        --order_data "$order_data" \
                                        --seed "$seed" \
                                        --load_with_llm "$llm" \
                                        --nr_samples "$nr_samples" \
                                        --pc_indep_test "chisq" \
                                        --pc_alpha 0.05 \
                                        --min_samples "$min_sample" \
                                        --compute_min_prob_threshold "$compute_min_prob_threshold" \
                                        --anti_tags "$anti_tags" \
                                        --remove_duplicates "$remove_duplicates" \
                                        --remove_singular_tags "$remove_singular" \
                                        --prior_on_weight "$prior" \
                                        --always_meek "$always_meek" \
                                        --redirect_existing_edges "$redirect" &
                                done
                            done
                            wait
                        fi
                    done
                done
            done
        done
    done
done

end_time=$(date +%s)
total_time=$((end_time - start_time))

echo "Total computation time: $total_time seconds"
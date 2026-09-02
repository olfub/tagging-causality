import itertools
import json
import os

import numpy as np
from scipy.stats import rankdata

# E5_run_eval_laplace.sh
parameters = {}
parameters["experimental_series"] = ["E5_main_evaluation_laplace"]
parameters["identifier"] = ["bnlearn_child", "bnlearn_earthquake", "bnlearn_insurance", "bnlearn_survey", "bnlearn_asia", "bnlearn_cancer", "bnlearn_alarm", "lucas", "bnlearn_hepar2", "bnlearn_win95pts", "bnlearn_hailfinder"]
parameters["order_data"] = ["random"]
parameters["seed"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
parameters["llm"] = ["openai--gpt-5.2", "anthropic--claude-opus-4.6", "google--gemini-3-pro-preview", "meta-llama--llama-3.3-70b-instruct", "qwen--qwen3.5-397b-a17b", "z-ai--glm-5", "minimax--minimax-m2.5"]
parameters["nr_samples"] = [10000]
parameters["pc_indep_test"] = ["chisq"]
parameters["pc_alpha"] = [0.05]
parameters["ges_indep_test"] = ["local_score_BDeu"]
parameters["min_samples"] = [1, 2]
parameters["compute_min_prob_threshold"] = [True]
parameters["min_prob_threshold_default"] = [0.5]
parameters["anti_tags"] = [False]
parameters["remove_duplicates"] = [True]
parameters["remove_singular_tags"] = [True, False]
parameters["prior_on_weight"] = [True, False]
parameters["always_meeks"] = [True, False]
parameters["redirect_existing_edges"] = [True, False]
parameters["redirecting_strategy"] = [0, 1]
parameters["min_prob_redirecting"] = [0.6]
parameters["include_current_edge_as_evidence"] = [True, False]
parameters["include_redirected_edges_in_edge_count"] = [True]

all_methods = ["true", "skel", "skel_v", "pc", "skel_v_meeks", "ges", "llm_on_pc", "llm_on_true_cpdag", "llm_on_ges", "llm_root_recursive", "typed_pc_naive", "typed_pc_maj", "typed_prop_pc", "typed_prop_ges", "tag_pc_1", "tag_pc_0", "tag_pc_0_on_skel_v", "tag_pc_0_on_ges"]
methods_to_consider = all_methods
paper_table = 0
if paper_table == 0:
    methods_to_consider = ["pc", "ges", "typed_pc_naive", "typed_pc_maj", "tag_pc_1", "tag_pc_0", "tag_pc_0_on_ges"]
elif paper_table == 1:
    methods_to_consider = ["skel_v_meeks", "tag_pc_0_on_skel_v", "tag_pc_0_on_ges", "llm_on_ges"]
elif paper_table == 2:
    methods_to_consider = ["pc", "ges", "typed_pc_naive", "typed_pc_maj", "typed_prop_pc", "typed_prop_ges", "tag_pc_1", "tag_pc_0", "tag_pc_0_on_ges", "llm_root_recursive", "llm_on_pc", "llm_on_ges"]
elif paper_table == 3:
    methods_to_consider = ["tag_pc_0", "tag_pc_0_on_ges"]
else:
    raise ValueError("Invalid paper_table value")

method_names = {"true": "True Graph", "skel": "Skeleton", "skel_v": "Skeleton (V)", "pc": "PC", "skel_v_meeks": "GT CPDAG", "ges": "GES", "llm_on_pc": "LLM on PC", "llm_on_true_cpdag": "LLM on True CPDAG", "llm_on_ges": "LLM on GES", "llm_root_recursive": "LLM Root Recursive", "typed_pc_naive": "Typed-PC (Naive)", "typed_pc_maj": "Typed-PC (Maj.)", "tag_pc_1": "Tagged-PC (AntiV)", "tag_pc_0": "Tagged-PC", "tag_pc_0_on_skel_v": "Tagging on GT CPDAG", "tag_pc_0_on_ges": "Tagged-GES", "typed_prop_pc": "PC + t-Propagation", "typed_prop_ges": "GES + t-Propagation"}
dataset_names = {"bnlearn_child": "Child", "bnlearn_earthquake": "Earthquake", "bnlearn_insurance": "Insurance",
                    "bnlearn_survey": "Survey", "bnlearn_asia": "Asia", "bnlearn_cancer": "Cancer",
                    "bnlearn_alarm": "Alarm", "lucas": "Lucas", "bnlearn_hepar2": "Hepar2", "bnlearn_win95pts": "Win95Pts", "bnlearn_hailfinder": "Hailfinder"}
llm_to_text = {"openai--gpt-5.2": "GPT 5.2", "anthropic--claude-opus-4.6": "Claude 4.6", "google--gemini-3-pro-preview": "Gemini 3 Pro", "meta-llama--llama-3.3-70b-instruct": "Llama 3.3", "qwen--qwen3.5-397b-a17b": "Qwen 3.5", "z-ai--glm-5": "GLM 5", "minimax--minimax-m2.5": "Minimax 2.5"}

parameter_to_text = {
    "llm": "LLM",
    "min_samples": "Min-Samples",
    "remove_singular_tags": "Fewer Tags",
    "prior_on_weight": "S. Prior",
    "always_meeks": "Always Meek",
    "redirect_existing_edges": "Redirect",
    "redirecting_strategy": "Strategy",
    "include_current_edge_as_evidence": "Include Edge",
    "include_redirected_edges_in_edge_count": "Include Redirected Edges",
    "order_data": "Order Data",
    "nr_samples": "Nr Samples",
    "pc_indep_test": "PC Indep Test",
    "pc_alpha": "PC Alpha",
    "ges_indep_test": "GES Indep Test",
    "compute_min_prob_threshold": "Compute Min Prob Threshold",
    "min_prob_threshold_default": "Min Prob Threshold Default",
    "anti_tags": "Anti Tags",
    "remove_duplicates": "Remove Duplicates",
    "min_prob_redirecting": "Min Prob Redirecting",
}


def safe_filename_label(label):
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in label)


def parameter_label(parameter_name):
    return parameter_to_text.get(parameter_name, parameter_name)


def method_label(method):
    return method_names.get(method, method)


def read_eval_csv(file):
    with open(file, "r") as file:
        csv_content = file.read()
        metrics = []
        models = []
        for line in csv_content.splitlines():
            model = line.split(",")[0]
            metric = line.split(",")[1:]
            metrics.append(metric)
            models.append(model)
        return np.array(metrics, dtype=float), models

def metrics_to_ranks(metrics):
    # metrics should be in the form (n_samples, n_metrics) where ranks are computed per metric
    # n_metrics = 7, where the first 4 are to be maximized and the last 3 to be minimized
    ranks = np.zeros_like(metrics)
    for i in range(metrics.shape[1]):
        if i % 7 < 4:
            ranks[:, i] = rankdata(metrics[:, i], method='min')
        else:
            ranks[:, i] = rankdata(-metrics[:, i], method='min')
    return ranks

def read_results(parameters):
    all_evals = {}
    assert len(parameters["experimental_series"]) == 1, "Can only read results for one experimental series at a time"
    experimental_series = parameters["experimental_series"][0]
    path = f"results/{experimental_series}"
    models = []
    already_had_redirect_with_this = []
    param_keys = list(parameters.keys())
    param_keys.remove("experimental_series")
    param_values = [parameters[key] for key in param_keys]
    for parameter_combination in itertools.product(*param_values):
        current_iter_params = dict(zip(param_keys, parameter_combination))
        identifier = current_iter_params["identifier"]
        order_data = current_iter_params["order_data"]
        seed = current_iter_params["seed"]
        llm = current_iter_params["llm"]
        nr_samples = current_iter_params["nr_samples"]
        pc_indep_test = current_iter_params["pc_indep_test"]
        pc_alpha = current_iter_params["pc_alpha"]
        ges_indep_test = current_iter_params["ges_indep_test"]
        min_samples = current_iter_params["min_samples"]
        compute_min_prob_threshold = current_iter_params["compute_min_prob_threshold"]
        min_prob_threshold_default = current_iter_params["min_prob_threshold_default"]
        anti_tags = current_iter_params["anti_tags"]
        remove_duplicates = current_iter_params["remove_duplicates"]
        remove_singular_tags = current_iter_params["remove_singular_tags"]
        prior_on_weight = current_iter_params["prior_on_weight"]
        always_meeks = current_iter_params["always_meeks"]
        redirect_existing_edges = current_iter_params["redirect_existing_edges"]
        redirecting_strategy = current_iter_params["redirecting_strategy"]
        min_prob_redirecting = current_iter_params["min_prob_redirecting"]
        include_current_edge_as_evidence = current_iter_params["include_current_edge_as_evidence"]
        include_redirected_edges_in_edge_count = current_iter_params["include_redirected_edges_in_edge_count"]
        before_redirect_params = (identifier, order_data, seed, llm, nr_samples, pc_indep_test, pc_alpha, ges_indep_test, min_samples, compute_min_prob_threshold, min_prob_threshold_default, anti_tags, remove_duplicates, remove_singular_tags, prior_on_weight, always_meeks)
        if before_redirect_params in already_had_redirect_with_this and redirect_existing_edges == False:
            continue
        elif redirect_existing_edges == False:
            already_had_redirect_with_this.append(before_redirect_params)
            # while the following paramters do not matter, these are the default values that determined where the results were saved
            redirecting_strategy = 1
            include_current_edge_as_evidence = False
            include_redirected_edges_in_edge_count = True

        fewer_tags_str = "00"
        result_path = f"{path}/{identifier}/{order_data}/{seed}/{pc_indep_test}_{pc_alpha}_{ges_indep_test}_{nr_samples}/{llm}/{anti_tags}_{remove_duplicates}_{remove_singular_tags}_{prior_on_weight}_{min_samples}_{compute_min_prob_threshold}_{min_prob_threshold_default}_{fewer_tags_str}/{always_meeks}_{redirect_existing_edges}_{redirecting_strategy}_{include_current_edge_as_evidence}_{include_redirected_edges_in_edge_count}_{min_prob_redirecting}/_tagging_alg0"
        eval_file = f"{result_path}/eval.csv"
        eval_result, temp_models = read_eval_csv(eval_file)  # assuming models is the same for all considered files
        if models == []:
            models = temp_models
        else:
            assert models == temp_models
        all_evals[parameter_combination] = eval_result
        if parameter_combination[0] in ["bnlearn_hepar2", "bnlearn_win95pts", "bnlearn_hailfinder"]:
            assert np.all(all_evals[parameter_combination][:, 2:4] == 0)
            all_evals[parameter_combination][:, 2:4] = np.nan
    return all_evals, param_keys, models

def save_full_csv(parameters, all_evals_original, param_keys, models):
    experimental_series = parameters["experimental_series"][0]
    # save all_evals to csv
    with open(f"results/{experimental_series}/_full_eval/all_evals.csv", "w") as file:
        config_str = ",".join(param_keys)
        metric_str = "SHD,SHD(double_for_anticausal),SID_min,SID_max,precision,recall,F1"
        file.write(f"{config_str},method,{metric_str}\n")
        for key, value in all_evals_original.items():
            line_base = ",".join([str(x) for x in key])
            for idx, model in enumerate(models):
                line_str = line_base
                line_str += f",{model},{','.join(map(str, value[idx]))}\n"
                file.write(line_str)

def filter_methods(all_evals, all_models, selected_methods):
    # filter out methods and only return actual evaluation values (not ranks)
    new_all_evals = {}
    for parameter_combination, eval_result in all_evals.items():
        new_eval_result = []
        for method in selected_methods:
            index = all_models.index(method)
            new_eval_result.append(eval_result[index])
        new_all_evals[parameter_combination] = np.array(new_eval_result)
    return new_all_evals

def flatten_datasets(all_evals):
    # flatten all datasets for the same configuration in the same order
    new_all_evals = {}
    for parameter_combination, eval_result in all_evals.items():
        dataset = parameter_combination[0]
        config = tuple(parameter_combination[1:])
        if new_all_evals.get(config) is None:
            new_all_evals[config] = []
        new_all_evals[config].append((dataset, eval_result))
    for config in new_all_evals:
        new_all_evals[config].sort(key=lambda x: x[0])
        new_all_evals[config] = np.concatenate([x[1] for x in new_all_evals[config]], axis=1)
    return new_all_evals

def average_parameters(all_evals, to_average, return_stds=False):
    # average over parameters
    new_all_evals = {}
    for parameter_combination, eval_result in all_evals.items():
        config = tuple([parameter_combination[i] for i in range(len(parameter_combination)) if i not in to_average])
        if new_all_evals.get(config) is None:
            new_all_evals[config] = []
        new_all_evals[config].append(eval_result)
    all_evals_means = {}
    for config in new_all_evals:
        all_evals_means[config] = np.average(new_all_evals[config], axis=0)
    if return_stds:
        all_evals_stds = {}
        for config in new_all_evals:
            all_evals_stds[config] = np.std(new_all_evals[config], axis=0)
        return all_evals_means, all_evals_stds
    else:
        return all_evals_means

def get_best_config_by_f1(evals, param_keys, method_index):
    data_index = param_keys.index("identifier")
    param_keys_without_data = [key for key in param_keys if key != "identifier"]
    # configs as list
    all_configs = list(evals.keys())
    # put configs into dictionary by dataset
    configs_per_dataset = {}
    for config in all_configs:
        conf_without_ds = config[:data_index] + config[data_index+1:]  # without dataset
        # -1 is the f1 score
        eval_score = evals[config][method_index, -1]
        if configs_per_dataset.get(conf_without_ds) is None:
            configs_per_dataset[conf_without_ds] = []
        configs_per_dataset[conf_without_ds].append(eval_score)
    # get average f1 score per config
    configs_per_dataset_avg = {}
    for conf_without_ds, scores in configs_per_dataset.items():
        configs_per_dataset_avg[conf_without_ds] = np.average(scores)

    # print best 10 configs
    method = methods_to_consider[method_index]
    configs_list = [(conf, score) for conf, score in configs_per_dataset_avg.items()]
    sorted_configs_list = sorted(configs_list, key=lambda x: x[1])
    print(f"Best 10 methods for {method}")
    for conf, score in sorted_configs_list[-10:]:
        print_best_config(conf, param_keys_without_data, method)  # enter the right method here
        print(score)
    top_10_configs = sorted_configs_list[-10:][::-1]
    best_configs_to_latex(top_10_configs, param_keys_without_data, f"results/{parameters['experimental_series'][0]}/_configs/best_configs_{method}.txt")
    print(f"End of best 10 methods for {method}")

    # table for best config per llm
    llm_results = []
    for llm in parameters["llm"]:
        llm_configs_list = [(conf, score) for conf, score in configs_per_dataset_avg.items() if conf[param_keys_without_data.index("llm")] == llm]
        llm_configs_list.sort(key=lambda x: x[1])
        best_llm_config = llm_configs_list[-1]
        llm_results.append((best_llm_config[0], best_llm_config[1]))
    llm_results.sort(key=lambda x: x[1], reverse=True)
    best_configs_to_latex(llm_results, param_keys_without_data, f"results/{parameters['experimental_series'][0]}/_configs/best_configs_per_llm_{method}.txt")

    # get best config
    best_configs = []
    best_score = 0
    for conf, score in configs_per_dataset_avg.items():
        if best_score == score:
            best_configs.append(conf)
        elif score > best_score:
            best_configs = [conf]
            best_score = score

    print (f"Best f1 score: {best_score}")
    return best_configs, configs_per_dataset_avg, param_keys_without_data


def best_configs_to_latex(configs_list, param_keys, file_path):
    latex_output = [
        "\\begin{tabular}{cccccccc|c}",
        "    \\toprule",
        "    LLM & Min-Samples & Fewer Tags & S. Prior & Always Meek & Redirect & Strategy & Include Edge & F\\textsubscript{1} \\\\",
        "    \\midrule"
    ]

    # Process the top 10 entries
    for best_conf, score in configs_list:
        # 1. Extraction using the param_keys mapping
        llm = best_conf[param_keys.index("llm")]
        llm = llm_to_text[llm]
        min_samples = best_conf[param_keys.index("min_samples")]
        remove_singular_tags = best_conf[param_keys.index("remove_singular_tags")]
        prior_on_weight = best_conf[param_keys.index("prior_on_weight")]
        always_meeks = best_conf[param_keys.index("always_meeks")]
        redirect = best_conf[param_keys.index("redirect_existing_edges")]
        strategy = best_conf[param_keys.index("redirecting_strategy")]
        include_edge = best_conf[param_keys.index("include_current_edge_as_evidence")]
        
        # 2. Logic for the dash override
        # If Redirect is False, following parameters are masked
        display_strategy = str(strategy) if redirect else "-"
        display_include = str(include_edge).capitalize() if redirect else "-"
        
        # 3. Formatting the row
        # .capitalize() ensures "True"/"False" matches your LaTeX example
        row = (
            f"    {llm} & "
            f"{min_samples} & "
            f"{str(remove_singular_tags).capitalize()} & "
            f"{str(prior_on_weight).capitalize()} & "
            f"{str(always_meeks).capitalize()} & "
            f"{str(redirect).capitalize()} & "
            f"{display_strategy} & "
            f"{display_include} & "
            f"{score:.4f} \\\\"
        )
        latex_output.append(row)

    # 4. Closing the table
    latex_output.append("    \\bottomrule")
    latex_output.append("\\end{tabular}")

    # Write to file
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(latex_output))
        print(f"Successfully saved table to {file_path}")
    except Exception as e:
        print(f"Failed to save file: {e}")


def best_config_sensitivity_to_text(best_conf, best_score, configs_per_dataset_avg, param_keys, method, file_path):
    lines = []
    lines.append(f"Method: {method}")
    lines.append("Best config:")
    for idx, param_name in enumerate(param_keys):
        current_value = best_conf[idx]
        if param_name == "llm":
            current_value = llm_to_text[current_value]
        lines.append(f"  - {parameter_label(param_name)}: {current_value}")
    lines.append(f"  - avg_f1: {best_score:.6f}")
    lines.append("")

    for idx, param_name in enumerate(param_keys):
        lines.append(f"Parameter: {parameter_label(param_name)}")
        lines.append("value,avg_f1,delta_vs_best")

        base_value = best_conf[idx]
        base_display = llm_to_text[base_value] if param_name == "llm" else base_value
        lines.append(f"{base_display},{best_score:.6f},{0.0:+.6f}")

        for value in parameters[param_name]:
            if value == base_value:
                continue
            candidate = list(best_conf)
            candidate[idx] = value
            candidate = tuple(candidate)
            if candidate not in configs_per_dataset_avg:
                redirect_off = best_conf[param_keys.index("redirect_existing_edges")] == False
                if param_name == "redirect_existing_edges" or (redirect_off and param_name in ("redirecting_strategy", "include_current_edge_as_evidence", "include_redirected_edges_in_edge_count")):
                    # these three parameters have no effect when redirect_existing_edges=False, so only one combo of them was ever run
                    continue
                else:
                    raise ValueError(f"Candidate config {candidate} not found in configs_per_dataset_avg")
            score = configs_per_dataset_avg[candidate]
            delta = score - best_score
            display_value = llm_to_text[value] if param_name == "llm" else value
            lines.append(f"{display_value},{score:.6f},{delta:+.6f}")
        lines.append("")

    with open(file_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def best_config_sensitivity_latex(best_conf, best_score, configs_per_dataset_avg, param_keys, method, file_path):
    rows = []
    for idx, param_name in enumerate(param_keys):
        if len(parameters[param_name]) <= 1:
            continue

        base_value = best_conf[idx]
        for value in parameters[param_name]:
            if value == base_value:
                continue
            candidate = list(best_conf)
            candidate[idx] = value
            candidate = tuple(candidate)
            if candidate not in configs_per_dataset_avg:
                redirect_off = best_conf[param_keys.index("redirect_existing_edges")] == False
                if param_name == "redirect_existing_edges" or (redirect_off and param_name in ("redirecting_strategy", "include_current_edge_as_evidence", "include_redirected_edges_in_edge_count")):
                    # these three parameters have no effect when redirect_existing_edges=False, so only one combo of them was ever run
                    continue
                else:
                    raise ValueError(f"Candidate config {candidate} not found in configs_per_dataset_avg")
            score = configs_per_dataset_avg[candidate]
            delta = score - best_score
            if np.isclose(delta, 0.0):
                continue
            display_value = llm_to_text[value] if param_name == "llm" else value
            rows.append((parameter_label(param_name), display_value, delta))

    rows.sort(key=lambda row: row[2])

    latex_lines = [
        "\\begin{tabular}{l|l|r}",
        f"\\textbf{{Parameter}} & \\textbf{{Value}} & \\textbf{{$\\Delta F_1$}} \\\\",
        "\\hline",
    ]
    current_param = None
    for param_name, value, delta in rows:
        if current_param is not None and param_name != current_param:
            latex_lines.append("\\hline")
        current_param = param_name
        latex_lines.append(f"{param_name} & {value} & {delta:+.6f} \\\\")
    latex_lines.append("\\end{tabular}")

    with open(file_path, "w", encoding="utf-8") as file:
        file.write("\n".join(latex_lines))


def print_metrics_nicely(metrics, methods):
    for i in range(metrics.shape[0]):
        method_string = (methods[i] + ":").ljust(16)
        metrics_string = ", ".join([f"{metrics[i, j]:.2f}" for j in range(metrics.shape[1])])
        print(f"{method_string} {metrics_string}")


def metrics_to_latex(file, metrics, methods, stds=None):
    # this method plots the average ranks over all datasets
    bf_values = np.zeros(metrics.shape[1])
    for i in range(metrics.shape[1]):
        bf_values[i] = np.min(metrics[:, i])
    with open(file, "w") as file:
        file.write("\\begin{tabular}{l|ccccccc}\n")
        file.write("& SHD & SHD\\textsubscript{double} & SID\\textsubscript{min} & SID\\textsubscript{max} & Precision & Recall & F\\textsubscript{1} \\\\\n")
        file.write(" & Ranks & Ranks & Ranks & Ranks & Ranks & Ranks & Ranks \\\\\n")
        file.write("\\hline\n")
        for i in range(metrics.shape[0]):
            method_str = method_names[methods[i]]
            file.write(f"{method_str} & ")
            for j in range(metrics.shape[1]):
                metric_str = "$"
                if metrics[i, j] == bf_values[j]:
                    metric_str += f"\\mathbf{{{metrics[i, j]:.2f}}}"
                else:
                    metric_str += f"{metrics[i, j]:.2f}"
                if stds is not None:
                    metric_str += f" {{\\scriptstyle \\pm {stds[i, j]:.2f}}}"
                metric_str += "$"
                if j == metrics.shape[1] - 1:
                    file.write(f"{metric_str} \\\\\n")
                else:
                    file.write(f"{metric_str} & ")
            if methods[i] == "ges" or methods[i] == "typed_pc_maj":
                file.write("\\hline\n")
        file.write("\\end{tabular}")


def metrics_to_latex_plus(file, data, datasets, methods, stds=None, hlines_after=None):
    # this methods plots all datasets individually
    with open(file, "w") as file:
        file.write("\\begin{tabular}{l|ccccccc}\n")
        file.write("\\textbf{Evaluation Results} & SHD & SHD\\textsubscript{double} & SID\\textsubscript{min} & SID\\textsubscript{max} & Precision & Recall & F\\textsubscript{1} \\\\\n")
        for dataset in datasets:
            file.write("\\hline \\hline\n")
            dataset_name = dataset_names[dataset]
            file.write(f"Dataset {dataset_name} & & & & & & & \\\\\n")
            file.write("\\hline\n")
            current_data = data[dataset]
            current_stds = stds[dataset] if stds is not None else None
            bf_values = np.zeros(current_data.shape[1])
            for i in range(current_data.shape[1]):
                if i < 4:
                    bf_values[i] = np.min(current_data[:, i])
                else:
                    bf_values[i] = np.max(current_data[:, i])
            for i in range(current_data.shape[0]):
                method_str = method_names[methods[i]]
                file.write(f"{method_str} & ")
                for j in range(current_data.shape[1]):
                    metric_str = "$"
                    value = current_data[i, j]
                    if np.isnan(value):
                        metric_str = "-"
                    elif value == bf_values[j]:
                        metric_str += f"\\mathbf{{{value:.2f}}}"
                    else:
                        metric_str += f"{value:.2f}"
                    if stds is not None and not np.isnan(value):
                        metric_str += f" {{\\scriptstyle \\pm {current_stds[i, j]:.2f}}}"
                    if not np.isnan(value):
                        metric_str += "$"
                    if j == current_data.shape[1] - 1:
                        file.write(f"{metric_str} \\\\\n")
                    else:
                        file.write(f"{metric_str} & ") 
                if hlines_after is not None and methods[i] in hlines_after:
                    file.write("\\hline\n")
        file.write("\\end{tabular}")


def check_best_configs_for_uniqueness(eval_data, param_keys, idents, midx, best_confs):
    data_index = param_keys.index("identifier")
    if len(best_confs) == 1:
        best_conf = best_confs[0]
    else:
        dataset_evals_temp = {ds : [] for ds in idents}
        for best_conf in best_confs:
            for eval in eval_data:
                eval_without_ds = eval[1:data_index] + eval[data_index+1:]
                if eval_without_ds == best_conf:
                    dataset_evals_temp[eval[data_index]].append(eval_data[eval][midx])
        for ds in dataset_evals_temp:
            assert len(dataset_evals_temp[ds]) == len(best_confs)
            first_res = dataset_evals_temp[ds][0]
            for res in dataset_evals_temp[ds][1:]:
                # just make sure, that the results for the different configs are actually the same (up to numerical issues)
                # and that we are not just lucky that they have the same average f1 score but actually differ in other metrics or something 
                assert np.allclose(first_res, res, equal_nan=True)  # if this turns out false, we have to think of what we want to do then
        best_conf = best_confs[0]
    return best_conf


def print_best_config(best_conf, param_keys, m):
    assert len(param_keys) == len(best_conf)
    print(f"Best config for {m}")
    llm = best_conf[param_keys.index("llm")]
    min_samples = best_conf[param_keys.index("min_samples")]
    remove_singular_tags = best_conf[param_keys.index("remove_singular_tags")]
    prior_on_weight = best_conf[param_keys.index("prior_on_weight")]
    always_meeks = best_conf[param_keys.index("always_meeks")]
    redirect_existing_edges = best_conf[param_keys.index("redirect_existing_edges")]
    redirecting_strategy = best_conf[param_keys.index("redirecting_strategy")]
    include_current_edge_as_evidence = best_conf[param_keys.index("include_current_edge_as_evidence")]
    include_redirected_edges_in_edge_count = best_conf[param_keys.index("include_redirected_edges_in_edge_count")]
    print(f"{llm}, min_samples: {min_samples}, remove_sing_tags: {remove_singular_tags}, prior: {prior_on_weight}, always_meeks: {always_meeks}, redirect_edges: {redirect_existing_edges}, strategy: {redirecting_strategy}, include_cur: {include_current_edge_as_evidence}, include_red: {include_redirected_edges_in_edge_count}")


all_evals_original, full_param_keys, models = read_results(parameters)
save_csv = False
if save_csv:
    save_full_csv(parameters, all_evals_original, full_param_keys, models)

all_evals_and_seeds = filter_methods(all_evals_original, models, methods_to_consider)
avg_over_index = [full_param_keys.index("seed")]
all_evals, all_evals_stds = average_parameters(all_evals_and_seeds, avg_over_index, return_stds=True)
param_keys = [key for index, key in enumerate(full_param_keys) if index not in avg_over_index]
os.makedirs(f"results/{parameters['experimental_series'][0]}/_configs", exist_ok=True)
os.makedirs(f"results/{parameters['experimental_series'][0]}/_result_tables", exist_ok=True)
os.makedirs(f"results/{parameters['experimental_series'][0]}/_full_eval", exist_ok=True)

if True:
    # get the best config by choosing the one with the best f1 score average (across seeds and datasets)
    assert len(parameters["experimental_series"]) == 1, "Can only read results for one experimental series at a time"
    experimental_series = parameters["experimental_series"][0]
    identifiers = parameters["identifier"]
    seeds = parameters["seed"]
    # dataset_evals = {ds : [] for ds in identifiers}
    best_config_by_method = {method : None for method in methods_to_consider}
    for method in methods_to_consider:
        method_idx = methods_to_consider.index(method)
        best_configs, configs_per_dataset_avg, param_keys_without_data_sens = get_best_config_by_f1(all_evals, param_keys, method_idx)
        best_config = check_best_configs_for_uniqueness(all_evals, param_keys, identifiers, method_idx, best_configs)
        best_config_by_method[method] = best_config

        if paper_table == 0:
            # hyperparameter sensitivity analysis for the chosen best config
            best_score_sens = max(configs_per_dataset_avg.values())
            best_config_sensitivity_to_text(
                best_config,
                best_score_sens,
                configs_per_dataset_avg,
                param_keys_without_data_sens,
                method,
                f"results/{experimental_series}/_configs/best_config_sensitivity_{safe_filename_label(method_label(method))}.txt",
            )
            best_config_sensitivity_latex(
                best_config,
                best_score_sens,
                configs_per_dataset_avg,
                param_keys_without_data_sens,
                method,
                f"results/{experimental_series}/_configs/best_config_differences_{safe_filename_label(method_label(method))}.tex",
            )

    param_keys_config = [key for key in param_keys if key != "identifier"]

    # save best_config to json
    for method in best_config_by_method:
        with open(f"results/{experimental_series}/_configs/_best_config_{method}.json", "w") as f:
            config_dict = {param_keys_config[i]: best_config_by_method[method][i] for i in range(len(param_keys_config))}
            json.dump(config_dict, f, indent=4)

    if paper_table == 1:
        # might have to run another config to generate these best config files
        methods_to_consider.remove("tag_pc_0_on_ges")
        methods_to_consider.remove("llm_on_ges")
        tag_ges_config = json.load(open(f"results/{experimental_series}/_configs/_best_config_tag_pc_0_on_ges.json", "r"))
        llm_ges_config = json.load(open(f"results/{experimental_series}/_configs/_best_config_llm_on_ges.json", "r"))
        best_config_by_method["tag_pc_0_on_skel_v"] = tuple([tag_ges_config[param] for param in param_keys_config])
        best_config_by_method["llm_on_true_cpdag"] = tuple([llm_ges_config[param] for param in param_keys_config])

    # ranks per seed
    ranks_per_seed = []
    average_rank_tables = []
    # consider seeds separately now
    for s in seeds:
        rank_tables_current_seed = []
        dataset_evals = {ds : [] for ds in identifiers}
        # collect the data for all seeds using their respective configs
        for method in methods_to_consider:
            method_idx = methods_to_consider.index(method)
            config = best_config_by_method[method]
            for eval in all_evals_and_seeds:
                # the eval config has identifier and seed, which conf does not have, so we need to remove those from eval to compare with conf
                eval_config = tuple(econf for (eparam, econf) in zip(full_param_keys, eval) if eparam not in ["identifier", "seed"])
                eval_seed = eval[full_param_keys.index("seed")]
                # this checks that the config (conf) matches the eval config
                # conf does not contain a seed, but eval has it, so this needs to match as well
                if config == eval_config and eval_seed == s:
                    data_idx = full_param_keys.index("identifier")
                    dataset_evals[eval[data_idx]].append(all_evals_and_seeds[eval][method_idx])
        for dataset in dataset_evals:
            assert len(dataset_evals[dataset]) == len(methods_to_consider)
            dataset_evals[dataset] = np.array(dataset_evals[dataset])
            rank_tables_current_seed.append(metrics_to_ranks(dataset_evals[dataset]))
        average_rank_tables.append(np.nanmean(np.array(rank_tables_current_seed), axis=0))
    all_average_ranks = np.average(np.array(average_rank_tables), axis=0)
    all_average_ranks_stds = np.std(np.array(average_rank_tables), axis=0)
    print("Average ranks overall:")
    print_metrics_nicely(all_average_ranks, methods_to_consider)
    if paper_table == 0:
        name = "all_ranks"
    elif paper_table == 1:
        name = "true_skeleton"
    elif paper_table == 2:
        name  = "all_ranks_plus"
    elif paper_table == 3:
        name = "tagging_only"
    metrics_to_latex(f"results/{experimental_series}/_result_tables/{name}.txt", all_average_ranks, methods_to_consider, stds=all_average_ranks_stds)

    # now per dataset, no ranks
    data_for_table = {}
    data_for_table_stds = {}
    data_idx = full_param_keys.index("identifier")
    for dataset in identifiers:
        for method in methods_to_consider:
            for eval in all_evals:
                eval_dataset = eval[data_idx]
                eval_config = tuple(econf for (eparam, econf) in zip(full_param_keys, eval) if eparam != "identifier")
                if eval_dataset == dataset and eval_config == best_config_by_method[method]:
                    data_for_table[(dataset, method)] = all_evals[eval][methods_to_consider.index(method)]
                    data_for_table_stds[(dataset, method)] = all_evals_stds[eval][methods_to_consider.index(method)]
    data_tables = {}
    data_tables_stds = {}
    for dataset in identifiers:
        data_tables[dataset] = np.array([data_for_table[(dataset, method)] for method in methods_to_consider])
        data_tables_stds[dataset] = np.array([data_for_table_stds[(dataset, method)] for method in methods_to_consider])
    split_table_into = 1
    hlines_after = None
    if paper_table == 0:
        name = "all_datasets"
        split_table_into = 2
    elif paper_table == 1:
        name = "true_skeleton_datasets"
    elif paper_table == 2:
        name = "all_datasets_plus"
        split_table_into = 3
        hlines_after = ["tag_pc_0_on_ges"]
    elif paper_table == 3:
        name = "tagging_only_datasets"
        split_table_into = 1
    if split_table_into == 1:
        datasets = ["bnlearn_cancer", "bnlearn_earthquake", "bnlearn_survey", "bnlearn_asia", "lucas", "bnlearn_child", "bnlearn_alarm", "bnlearn_insurance", "bnlearn_hailfinder", "bnlearn_hepar2", "bnlearn_win95pts"]
        metrics_to_latex_plus(f"results/{experimental_series}/_result_tables/{name}.txt", data_tables, datasets, methods_to_consider, stds=data_tables_stds, hlines_after=hlines_after)
    elif split_table_into == 2:
        datasets = ["bnlearn_cancer", "bnlearn_earthquake", "bnlearn_survey", "bnlearn_asia", "lucas"]
        metrics_to_latex_plus(f"results/{experimental_series}/_result_tables/{name}.txt", data_tables, datasets, methods_to_consider, stds=data_tables_stds, hlines_after=hlines_after)
        datasets = ["bnlearn_child", "bnlearn_alarm", "bnlearn_insurance", "bnlearn_hailfinder", "bnlearn_hepar2", "bnlearn_win95pts"]
        metrics_to_latex_plus(f"results/{experimental_series}/_result_tables/{name}_2.txt", data_tables, datasets, methods_to_consider, stds=data_tables_stds, hlines_after=hlines_after)
    elif split_table_into == 3:
        datasets = ["bnlearn_cancer", "bnlearn_earthquake", "bnlearn_survey", "bnlearn_asia"]
        metrics_to_latex_plus(f"results/{experimental_series}/_result_tables/{name}.txt", data_tables, datasets, methods_to_consider, stds=data_tables_stds, hlines_after=hlines_after)
        datasets = ["lucas", "bnlearn_child", "bnlearn_alarm", "bnlearn_insurance"]
        metrics_to_latex_plus(f"results/{experimental_series}/_result_tables/{name}_2.txt", data_tables, datasets, methods_to_consider, stds=data_tables_stds, hlines_after=hlines_after)
        datasets = ["bnlearn_hailfinder", "bnlearn_hepar2", "bnlearn_win95pts"]
        metrics_to_latex_plus(f"results/{experimental_series}/_result_tables/{name}_3.txt", data_tables, datasets, methods_to_consider, stds=data_tables_stds, hlines_after=hlines_after)
    else:
        raise ValueError("Invalid split option")

print("Done")
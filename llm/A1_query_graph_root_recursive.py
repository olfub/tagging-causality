from os import mkdir
from pathlib import Path
import json
import re
import networkx as nx

from tagging_causality.llm.datasets import bn_datasets, bn_topics
from tagging_causality.llm.bfs_ds_descs import BFS_VAR_NAMES_AND_DESC

from tagging_causality.llm.llms_router import LLMProvider

llms = [
    "openai--gpt-5.2",
    "anthropic--claude-opus-4.6",
    "google--gemini-3-pro-preview",
    "meta-llama--llama-3.3-70b-instruct",
    "qwen--qwen3.5-397b-a17b",
    "z-ai--glm-5",
    "minimax--minimax-m2.5",
]

save_path = Path("../queries/raw/root_recursive")
skip_existing = True

final_adj_path = Path("../queries/processed/root_recursive")

key_dir = Path("../api_keys/")


template_name = "root_recursive"

system_prompt = "You are a helpful assistant for causal reasoning."
# figure out root variables
root_prompt_template = \
    "You are a helpful assistant to a $TOPIC expert. The following factors are key variables related to $TOPIC which " +\
    "have various causal effects on each other. Our goal is to construct a causal graph between these variables:\n\n"+\
    "$VARLINES\n\n"+\
    "Now you are going to use the data to construct a causal graph. You will start with identifying the variable(s) "+\
    "that are unaffected by any other variables. Think step by step. Then, provide your final answer (comma separated "+\
    "variable keys only) within the tags <Answer>...</Answer>."""

# identify effects
effects_prompt_template = [
    "$VARLINES\n\n"+\
    "Given $INDEPVARS $ISARE not affected by any other variable and the following causal relationships:\n"
    "$RELATIONS\n\n" #A causes B, C, D\nC causes D, E\n...
    "Select the variables that are caused by $CURRENTVISITNODE. Think step by step. Then, provide your final answer "
    "(comma separated variable keys only) within the tags <Answer>...</Answer>."][0]


def assemble_messages(prompt_template, variables):
    for var_name, var_value in variables.items():
        prompt_template = prompt_template.replace(f"${var_name}", var_value)
    prompt = prompt_template

    if "$" in prompt:
        raise RuntimeError(f"Found '$' in prompt. Was there a variable not replaced in the template? Provided keys: {list(variables.keys())}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    return messages


def main():
    provider = LLMProvider(key_dir=key_dir)

    save_path.mkdir(exist_ok=True, parents=True)
    final_adj_path.mkdir(exist_ok=True, parents=True)

    del bn_datasets["win95pts"] # goes out of context

    for llm_name in llms:
        llm = provider.get_interface(llm_name)

        for ds_name, ds_variables in bn_datasets.items():
            spec_save_path = save_path / ds_name / llm_name
            spec_save_path.mkdir(exist_ok=True, parents=True)

            graph_path = f"{llm_name}_{template_name}_{ds_name}_adj"
            graph_save_loc = final_adj_path / f"{graph_path}.txt"
            var_order_save_loc = final_adj_path / f"{graph_path}_var_names.txt"
            print(">>", graph_path)
            if skip_existing and graph_save_loc.exists():
                pass

            topic = bn_topics[ds_name]

            # take better var descriptions if available
            if ds_name in BFS_VAR_NAMES_AND_DESC:
                ds_variables = BFS_VAR_NAMES_AND_DESC[ds_name]
            ds_variables_lower = {k.lower():v for k,v in ds_variables.items()}
            ds_variables_normal = list(ds_variables.keys())

            #adj = np.zeros((len(ds_variables_lower), len(ds_variables_lower)))
            graph = nx.DiGraph()
            graph.add_nodes_from(list(ds_variables_lower.keys()))
            #graph.is_directed_acyclic_graph()

            effects_list = {}

            def get_varlines(var_dict):
                return "\n".join([f"<{key}>: {val}" for key, val in var_dict.items()])

            # query root nodes
            query_name = f"{llm_name}_{template_name}_{ds_name}_root"
            save_loc = spec_save_path / f"{query_name}_roots.txt"
            print("ROOT>>", query_name)
            if skip_existing and save_loc.exists():
                # load variable names
                response = save_loc.read_text()
                queue = json.loads(response)
                roots = queue.copy()
            else:
                vars = {
                    "TOPIC": topic,
                    "VARLINES": get_varlines(ds_variables_lower)
                }
                messages = assemble_messages(root_prompt_template, vars)
                #print("root", messages)

                #response = llm.query(messages, max_tokens=None)
                max_tokens = llm._max_tokens if llm_name == "qwen--qwen3.5-397b-a17b--hf" else 4*llm._max_tokens
                response = llm.batch_queries([messages], max_tokens=max_tokens)[0]
                if isinstance(response, Exception):
                    raise response
                (spec_save_path / f"{query_name}_resp.txt").write_text(response)

                pattern = r"<Answer>(.*?)<\/Answer>"
                match = re.search(pattern, response, re.DOTALL)
                if match:
                    queue = []
                    content = match.group(1)
                    variable_cands = content.split(",")

                    for cand in variable_cands:
                        cand = cand.strip().lower()
                        if cand.startswith("<") and cand.endswith(">"):
                            cand = cand[1:-1]
                        idx = ds_variables_lower.get(cand, None)
                        if idx is None:
                            print("skipping", cand)
                            pass
                        queue.append(cand)

                    if len(queue) == 0:
                        raise RuntimeError("No roots identified.")
                else:
                    raise RuntimeError(f"<Answer> tag was not found. Check {query_name}_resp.txt and write {query_name}_roots.txt")
                roots = queue.copy()
                save_loc.write_text(json.dumps(queue))

            def build_relations(effects_list):
                if len(effects_list) == 0:
                    return ["Currently none."]
                return [f"{cause} causes {', '.join(effects)}" for cause, effects in effects_list.items()]

            # expansion
            visited = set()
            while len(queue) != 0:
                target = queue.pop(0)
                visited.add(target)
                query_name = f"{llm_name}_{template_name}_{ds_name}_{target}"
                save_loc = spec_save_path / f"{query_name}.txt"
                if skip_existing and save_loc.exists():
                    # load variable names
                    response = save_loc.read_text()
                    effects = json.loads(response)
                else:
                    vars = {
                        "VARLINES": get_varlines(ds_variables_lower),
                        "INDEPVARS": ", ".join(roots),
                        "ISARE": "is" if len(roots)==1 else "are",
                        "CURRENTVISITNODE": target,
                        "RELATIONS": "\n".join(build_relations(effects_list)) #"A causes B, C, D\nC causes D, E\n..."
                    }
                    print(f"{ds_name}_{target}")
                    messages = assemble_messages(effects_prompt_template, vars)
                    print("effects", messages)

                    #response = llm.query(messages, max_tokens=None)
                    max_tokens = llm._max_tokens if llm_name == "qwen--qwen3.5-397b-a17b--hf" else 4 * llm._max_tokens
                    response = llm.batch_queries([messages], max_tokens=max_tokens)[0]
                    if isinstance(response, Exception):
                        raise response
                    (spec_save_path / f"{query_name}_resp.txt").write_text(response)

                    pattern = r"<Answer>(.*?)<\/Answer>"
                    match = re.search(pattern, response, re.DOTALL)
                    if match:
                        effects = []
                        content = match.group(1)
                        if content.strip() != "":
                            variable_cands = content.split(",")

                            print("cands", variable_cands)
                            for cand in variable_cands:
                                cand = cand.strip().lower()
                                if cand.startswith("<") and cand.endswith(">"):
                                    cand = cand[1:-1]
                                if cand.strip() == "":
                                    continue
                                idx = ds_variables_lower.get(cand, None)
                                if idx is None:
                                    print("skipping", cand)
                                    continue
                                effects.append(cand)
                    else:
                        handled = False
                        #gpt-4-0613_root_recursive_child_sick_resp
                        if response.startswith("Without any provided causal relationships"):
                            effects = []
                            handled=True
                        if not handled:
                            raise RuntimeError(
                                f"<Answer> tag was not found. Check {query_name}_resp.txt and write {query_name}.txt")
                    save_loc.write_text(json.dumps(effects))

                local_list = []
                for effect in effects:
                    if (effect not in visited) and (effect not in queue):
                        queue.extend(effects)

                    graph.add_edge(target, effect)
                    if not nx.is_directed_acyclic_graph(graph):
                        graph.remove_edge(target, effect)
                    else:
                        local_list.append(effect)
                if len(local_list) != 0:
                    effects_list[target] = local_list

            adj = nx.to_numpy_array(graph, nodelist=list(ds_variables_lower.keys())).tolist()
            # entry [i,j] corresponds to edge from i to j
            graph_save_loc.write_text(json.dumps(adj))
            var_order_save_loc.write_text(json.dumps(ds_variables_normal))

    print("done.")


if __name__ == "__main__":
    main()

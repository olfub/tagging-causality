from pathlib import Path
import json

from tagging_causality.llm.datasets import bn_datasets

llms = [
    "openai--gpt-5.2",
    "anthropic--claude-opus-4.6",
    "google--gemini-3-pro-preview",
    "meta-llama--llama-3.3-70b-instruct",
    "qwen--qwen3.5-397b-a17b",
    "z-ai--glm-5",
    "minimax--minimax-m2.5",
]

save_path = Path("../queries/processed/pairwise_completion")

load_path = Path("../llm_edge_prediction/")
variants = ["true_skel", "pc", "ges"]
load_answer_path = Path("../queries/raw/pairwise_completion")


template_name = "pairwise"


def load_required_pairs(dataset_name):
    pairs = set()
    for variant in variants:
        filename = f"{dataset_name}.txt" if dataset_name == "lucas" else f"bnlearn_{dataset_name}.txt"
        with (load_path / variant / filename).open("r") as f:
            rawlines = f.readlines()
        for line in rawlines:
            a,b = line.strip().split(" -- ")
            pairs.add((a,b))
    return pairs


def main():

    save_path.mkdir(exist_ok=True, parents=True)

    for llm_name in llms:
        for ds_name, ds_variables in bn_datasets.items():
            # TODO a bit risky, as this might get out of sync with the query script. Just collect all "_{ds_name}___*" files would be better.
            required_pairs = load_required_pairs(ds_name)

            pairs = {}
            no_answers = []
            for vara, varb in required_pairs:
                query_name = f"{llm_name}__{template_name}__{ds_name}___{vara}__{varb}"
                load_loc = load_answer_path / f"{query_name}.txt"

                with load_loc.open("r") as f:
                    raw_answer = f.read()

                    if "[[FINISHREASON]]" in raw_answer:
                        _, finish_reason = raw_answer.split("[[FINISHREASON]]")
                        print(f"[{query_name}] Finish reason: {finish_reason}")

                    num_answers = 0
                    answer = None
                    if "<Answer>A</Answer>" in raw_answer:
                        num_answers += 1
                        answer = "AB"
                    if "<Answer>B</Answer>" in raw_answer:
                        num_answers += 1
                        answer = "BA"
                    if num_answers == 1:
                        pairs[f"{vara}__{varb}"] = answer
                    else:
                        no_answers.append(f"{vara}__{varb}__{num_answers}")

                with (save_path / f"{llm_name}_{ds_name}.json").open("w+") as f:
                    json.dump(pairs, f, sort_keys=True)
                with (save_path / f"{llm_name}_{ds_name}_missing.json").open("w+") as f:
                    json.dump(no_answers, f, sort_keys=True)
    print("done.")


if __name__ == "__main__":
    main()

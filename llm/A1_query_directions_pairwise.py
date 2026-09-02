from pathlib import Path

from tagging_causality.llm.datasets import bn_datasets, bn_topics

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

save_path = Path("../queries/raw/pairwise_completion")
skip_existing = True

load_path = Path("../llm_edge_prediction/")
variants = ["true_skel", "pc", "ges"]


key_dir = Path("../api_keys/")


template_name = "pairwise"

# prompt as in Kiciman et al. "3.1 Pairwise causal edge inference: Inferring causal direction among variable pairs"/Figure 3.
system_prompt = "You are a helpful assistant for causal reasoning."
pairwise_prompt_template = \
    "You are a helpful assistant to a $TOPIC expert. Which cause-and-effect relationship is more likely?\n" + \
    "A. $VARA causes $VARB.\n" + \
    "B. $VARB causes $VARA.\n" + \
    "Let's work this out in a step by step way to be sure that we have the right answer. Then provide your final answer within the tags <Answer>A/B</Answer>."



def assemble_messages(prompt_template, variables):
    for var_name, var_value in variables.items():
        prompt_template = prompt_template.replace(f"${var_name}", var_value)
    prompt = prompt_template

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    return messages


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
    provider = LLMProvider(key_dir=key_dir)

    save_path.mkdir(exist_ok=True, parents=True)

    for llm_name in llms:
        llm = provider.get_interface(llm_name)
        max_concurrent_requests = llm.limiter.get_max_concurrent_requests()

        batch = []

        def query_batch(batch):
            print(f"querying {len(batch)}")

            results = llm.batch_queries([req["messages"] for req in batch])

            errs = []
            for req, res in zip(batch, results):
                if isinstance(res, Exception):
                    errs.append((req, res))
                else:
                    req["save_loc"].write_text(res)
            if len(errs) != 0:
                print("ERRORS")
                print(errs[0][0], ":")
                print(errs[0][1])
                #TODO upgrade to python 3.11 and use ExceptionGroup
                raise RuntimeError(errs)


        for ds_name, ds_variables in bn_datasets.items():
            topic = bn_topics[ds_name]

            required_pairs = load_required_pairs(ds_name)
            for vara, varb in required_pairs:
                query_name = f"{llm_name}__{template_name}__{ds_name}___{vara}__{varb}"
                save_loc = save_path / f"{query_name}.txt"
                #print(">>", query_name)
                if skip_existing and save_loc.exists():
                    continue

                vara_cap = ds_variables[vara]
                varb_cap = ds_variables[varb]
                vars = {
                    "TOPIC": topic,
                    "VARA": vara_cap,
                    "VARB": varb_cap
                }
                print(f"{query_name} {vara_cap}_{varb_cap}")
                messages = assemble_messages(pairwise_prompt_template, vars)

                batch.append({
                    "save_loc": save_loc,
                    "messages": messages,
                })

                if len(batch) == max_concurrent_requests:
                    query_batch(batch)
                    batch=[]
        if len(batch) != 0:
            query_batch(batch)
    print("done.")


if __name__ == "__main__":
    main()

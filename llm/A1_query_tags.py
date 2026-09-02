from pathlib import Path

from tagging_causality.llm.datasets import bn_datasets
from tagging_causality.llm.llms_router import LLMProvider

llms = [
    "openai--gpt-5.6-sol",
    "openai--gpt-5.5",
    "openai--gpt-5.4",
    "openai--gpt-5.2",
    "anthropic--claude-opus-5",
    "anthropic--claude-opus-4.8",
    "anthropic--claude-opus-4.7",
    "anthropic--claude-opus-4.6",
    "google--gemini-3-pro-preview",
    "meta-llama--llama-3.3-70b-instruct",
    "qwen--qwen3.8-max",
    "qwen--qwen3.7-max",
    "qwen--qwen3.6-max-preview",
    "qwen--qwen3.5-397b-a17b",
    "minimax--minimax-m2.5",
    "z-ai--glm-5"
]

save_path = Path("../queries/raw/taggingsets")
skip_existing = True


key_dir = Path("../api_keys/")


system_prompt = "You are an expert in annotating variables to provide additional information that helps to support a causal discovery algorithm."

tagging_prompt_template = """A tag is a single word or short phrase that describes a variable. Tags should be general enough to be applicable to multiple variables but specific enough to identify differences between similar variables. Tags will be used to identify causal directions between variables. Therefore, the individual sets of tags per variable should be discriminative enough to inform the algorithm. Variables can have multiple tags.
Consider the following variables: $VARIABLES.

Please generate a list of tags that can be assigned to one or multiple variables. Generate the number of tags necessary to strike a good balance between expressivity and specificity. Avoid duplicate tags that contain the same set of variables. Reply with one line per tag, where each line starts with the name of the tag, followed by a colon, and then a comma-separated list of variables that have that tag. The output should be machine parsable. For that reason, do not include any explanations or additional comments."""

typing_prompt_template = """A type is a single word or short phrase that describes a variable. Types should be general enough to be applicable to multiple variables but specific enough to identify differences between similar variables. Types will be used to identify causal directions between variables. Therefore, the individual types should be discriminative enough to inform the algorithm. Variables are assigned to a single type only.
Consider the following variables: $VARIABLES.

Please generate a list of types that can be assigned to one or multiple variables. Generate the number of types necessary to strike a good balance between expressivity and specificity. Reply with one line per type, where each line starts with the name of the type, followed by a colon, and then a comma-separated list of variables that belong to that type. Make sure that no variable appears in more than one the lists. The output should be machine parsable. For that reason, do not include any explanations or additional comments."""

templates = {
    "tag": tagging_prompt_template,
    #"type": typing_prompt_template
}
run_no = 0


def assemble_messages(prompt_template, variables):
    variables_str = ", ".join(variables)

    prompt = prompt_template.replace("$VARIABLES", variables_str)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    return messages


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
                raise errs[0][1]
                raise RuntimeError(errs)

        for template_name, template_str in templates.items():
            for ds_name, ds_variables in bn_datasets.items():
                query_name = f"{llm_name}__{template_name}__{ds_name}___{run_no}"
                save_loc = save_path / f"{query_name}.txt"
                if skip_existing and save_loc.exists():
                    print("SKIP>>", query_name)
                    continue
                print(">>", query_name)

                var_strs = ds_variables.values()
                messages = assemble_messages(template_str, var_strs)

                batch.append({
                    "save_loc": save_loc,
                    "messages": messages,
                })

                if len(batch) == max_concurrent_requests:
                    query_batch(batch)
                    batch=[]
        if len(batch) != 0:
            query_batch(batch)
            batch = []
    print("done.")


if __name__ == "__main__":
    main()

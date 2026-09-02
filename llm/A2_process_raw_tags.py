from pathlib import Path

from datasets import bn_datasets


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

raw_path = Path("../queries/raw/taggingsets")
processed_path = Path("../queries/processed/taggingsets")

templates = ["tag", "type"]
run_no=0

def main():
    processed_path.mkdir(exist_ok=True, parents=True)

    for llm_name in llms:
        for template_name in templates:
            for ds_name, ds_variables in bn_datasets.items():
                query_name = f"{llm_name}__{template_name}__{ds_name}___{run_no}"
                raw_loc = raw_path / f"{query_name}.txt"
                processed_loc = processed_path / f"{query_name}.txt"
                #print(f"processing {query_name}")

                if not raw_loc.exists():
                    print(f"[{query_name}] Does not exist. Skipping.")
                    continue
                raw_text = raw_loc.read_text()
                raw_text = raw_text.split("</think>")[-1].strip() # only take the text after thinking (if the model was thinking)

                if "[[FINISHREASON]]" in raw_text:
                    raw_text, finish_reason = raw_text.split("[[FINISHREASON]]")
                    print(f"[{query_name}] Finish reason: {finish_reason}")
                    raw_text = raw_text.strip()

                save_strs = list(ds_variables.keys())
                var_strs = list(ds_variables.values())

                spellings = {var_name.lower():var_name for var_name in var_strs}

                lines = raw_text.split("\n")
                processed_lines = []
                for line in lines:
                    tag_name, var_names = line.split(":")
                    tag_name = tag_name.strip()

                    processed_var_names = []
                    for var_name in var_names.split(","):
                        var_name_x = var_name.strip()
                        var_name = spellings.get(var_name_x.lower(), None)  # get correct capitalisation of variable (might be altered by some LLMs)
                        if var_name is None:
                            print(f"[{query_name}] {var_name_x} is not a valid variable {var_strs}. Skipping.")
                            continue
                        processed_var_name = save_strs[var_strs.index(var_name)]  # replace text var name with BN var name
                        processed_var_names.append(processed_var_name)

                    processed_line = f"{tag_name}:{','.join(processed_var_names)}"
                    processed_lines.append(processed_line)

                processed_text = "\n".join(processed_lines)

                processed_loc.write_text(processed_text)
    print("done.")


if __name__ == "__main__":
    main()

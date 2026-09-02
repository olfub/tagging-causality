from pathlib import Path
import json

from tagging_causality.llm.variations.tags_per_var_budget import assemble_tags_per_var_budget_templates
from tagging_causality.llm.variations.prompt_decomposition import assemble_prompt_decomposition_templates
from tagging_causality.llm.variations.paraphrases import assemble_paraphrasing_templates

from datasets import bn_datasets


llms = [
    "anthropic--claude-opus-4.6",
]

raw_path = Path("../queries/raw/taggingsets_variation")
processed_path = Path("../queries/processed/taggingsets_variation")

templates = list({
    **assemble_prompt_decomposition_templates(),
    **assemble_paraphrasing_templates(),
    **assemble_tags_per_var_budget_templates(),
}.keys())
run_no=0

def main():
    processed_path.mkdir(exist_ok=True, parents=True)
    (processed_path / ".." / "taggingsets_variation_groups.txt").write_text(json.dumps(templates, indent=2))
    processed_instances = []

    for llm_name in llms:
        for template_name in templates:
            for ds_name, ds_variables in bn_datasets.items():
                query_name = f"{llm_name}__{template_name}__{ds_name}___{run_no}"
                processed_instances.append(query_name)
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
                    try:
                        tag_name, var_names = line.split(":", 1)
                        var_names = var_names.replace(":", ",")
                    except Exception:
                        print(f"Error in {query_name}")
                        raise
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
    (processed_path / ".." / "taggingsets_variation_list.txt").write_text(json.dumps(processed_instances, indent=2))
    print("done.")


if __name__ == "__main__":
    main()

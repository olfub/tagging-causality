
prompt_decomposition = {
    "tag_definition": "A tag is a single word or short phrase that describes a variable. ",
    "specificityTradeoff": "Tags should be general enough to be applicable to multiple variables but specific enough to identify differences between similar variables. ",
    "downstreamPurpose": "Tags will be used to identify causal directions between variables. ",
    "discriminativeness": "Therefore, the individual sets of tags per variable should be discriminative enough to inform the algorithm. ",
    "multipleTags": "Variables can have multiple tags.",
    "TAGS_AVAILABLE": "\nConsider the following variables: $VARIABLES.\n\nPlease generate a list of tags that can be assigned to one or multiple variables. ",
    "tag_balance": "Generate the number of tags necessary to strike a good balance between expressivity and specificity. ",
    "noDuplicated": "Avoid duplicate tags that contain the same set of variables. ",
    "RESPONSE_FORMAT": "Reply with one line per tag, where each line starts with the name of the tag, followed by a colon, and then a comma-separated list of variables that have that tag. The output should be machine parsable. For that reason, do not include any explanations or additional comments."
}

required_keys = ["TAGS_AVAILABLE", "RESPONSE_FORMAT"]
vary_keys = [k for k in prompt_decomposition.keys() if k not in required_keys]

def assemble_prompt_decomposition_templates():
    templates = {}
    for excluded_key in vary_keys:
        prompt_template = "".join([prompt_decomposition[k] for k in prompt_decomposition.keys() if k != excluded_key])
        templates[f"prompt_decomposition_{excluded_key}"] = prompt_template
    return templates


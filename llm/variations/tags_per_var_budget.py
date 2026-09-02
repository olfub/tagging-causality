
USER_PROMPT_PER_VAR_K = """A tag is a single word or short phrase that describes a variable. Tags should be general enough to be applicable to multiple variables but specific enough to identify differences between similar variables. Tags will be used to identify causal directions between variables. Therefore, the individual sets of tags per variable should be discriminative enough to inform the algorithm. Every variables MUST be assigned exactly $NUM_TAGS tags.
Consider the following variables: $VARIABLES.

Please generate a list of tags that can be assigned to one or multiple variables. Generate the number of tags necessary to strike a good balance between expressivity and specificity. Avoid duplicate tags that contain the same set of variables. Reply with one line per tag, where each line starts with the name of the tag, followed by a colon, and then a comma-separated list of variables that have that tag. The output should be machine parsable. For that reason, do not include any explanations or additional comments."""

def assemble_tags_per_var_budget_templates():
    # per-variable budget: "each variable must have exactly K tags" (absolute)
    K_PER_VAR = [2, 3, 4, 6, 8, 10]
    return {f"budget_tags_per_var_{k}": USER_PROMPT_PER_VAR_K.replace("$NUM_TAGS", str(k)) for k in K_PER_VAR}


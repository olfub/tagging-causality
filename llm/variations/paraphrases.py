
# Register/verbosity: a bullet list (basically matching current prompt), a terse bullet point list (reducing to the minimum) and an inflated verbose version with redundant restatement.
REGISTER = {
    "bullets": """- A tag is a single word or short phrase that describes a variable.
- Tags should be general enough to be applicable to multiple variables, but specific enough to identify differences between similar variables.
- Tags will be used to identify causal directions between variables. The individual sets of tags per variable should therefore be discriminative enough to inform the algorithm.
- Variables can have multiple tags.
- The variables are: $VARIABLES.
- Generate a list of tags that can be assigned to one or multiple variables.
- Generate the number of tags necessary to strike a good balance between expressivity and specificity.
- Avoid duplicate tags that contain the same set of variables.
- Reply with one line per tag: the name of the tag, a colon, then a comma-separated list of variables that have that tag.
- The output should be machine parsable. Do not include any explanations or additional comments.""",

    "terse": """- Tag = word or short phrase describing a variable.
- General enough for several variables, specific enough to separate similar ones.
- Purpose: identifying causal directions. Tag sets per variable must be discriminative.
- Multiple tags per variable allowed.
- Variables: $VARIABLES.
- Balance expressivity and specificity. No two tags with the same variable set.
- Format: one line per tag, `tag: var1, var2, ...`. Machine parsable. No other text.""",

    "verbose": """- A tag is a single word or a short phrase that describes a variable. That is, each tag is a brief textual label, consisting of one word or at most a few words, which characterizes some property of the variable it is assigned to.
- Tags should be general enough to be applicable to multiple variables. At the same time, tags should be specific enough to identify differences between similar variables. In other words, a tag should neither be so broad that it applies indiscriminately to every variable, nor so narrow that it applies to only one variable and captures nothing shared.
- Tags will be used to identify causal directions between variables. Because the tags serve this downstream purpose, the individual set of tags assigned to each variable should be discriminative enough to inform the algorithm. That is, two different variables should, wherever possible, receive different sets of tags, so that the algorithm can tell them apart.
- Variables can have multiple tags. There is no requirement that a variable be assigned only a single tag, and there is no requirement that the tags form a partition of the variables.
- Consider the following variables, which are the variables to be annotated: $VARIABLES.
- Please generate a list of tags that can be assigned to one or multiple variables. Each tag in the list should be accompanied by the variables to which it applies.
- Generate the number of tags necessary to strike a good balance between expressivity and specificity. Do not generate so few tags that the annotation becomes uninformative, and do not generate so many tags that the annotation becomes redundant.
- Avoid duplicate tags that contain the same set of variables. If two candidate tags would apply to exactly the same set of variables, they are redundant with one another, and only one of them should be kept.
- Reply with one line per tag, where each line starts with the name of the tag, followed by a colon, and then a comma-separated list of variables that have that tag. Each line therefore has the form `tag: variable, variable, ...`.
- The output should be machine parsable. For that reason, do not include any explanations or additional comments, and do not include any introductory or concluding text, headers, numbering, or formatting beyond the specified line format."""
}


# Politeness: one very polite level and 2 stricter ones: demanding and strongly imperative.
POLITENESS = {
    "polite": """A tag is a single word or short phrase that describes a variable. Ideally, tags would be general enough to be applicable to multiple variables but specific enough to identify differences between similar variables. The tags will be used to identify causal directions between variables, so it would be very helpful if the individual sets of tags per variable were discriminative enough to inform the algorithm. Variables may of course have multiple tags.
Consider the following variables: $VARIABLES.

If you would be so kind, could you please generate a list of tags that can be assigned to one or multiple variables? Please feel free to generate whatever number of tags you consider necessary to strike a good balance between expressivity and specificity. It would be much appreciated if duplicate tags containing the same set of variables could be avoided. When you are ready, please reply with one line per tag, where each line starts with the name of the tag, followed by a colon, and then a comma-separated list of variables that have that tag. Since the output needs to be machine parsable, we would kindly ask you to omit any explanations or additional comments. Thank you very much for your help.""",

    "demanding": """A tag is a single word or short phrase that describes a variable. Tags must be general enough to be applicable to multiple variables but specific enough to identify differences between similar variables. Tags will be used to identify causal directions between variables. The individual sets of tags per variable must therefore be discriminative enough to inform the algorithm. Variables can have multiple tags.
Consider the following variables: $VARIABLES.

Generate a list of tags that can be assigned to one or multiple variables. You need to generate the number of tags required to strike a good balance between expressivity and specificity. Duplicate tags that contain the same set of variables are not acceptable. Your reply must contain one line per tag, where each line starts with the name of the tag, followed by a colon, and then a comma-separated list of variables that have that tag. The output has to be machine parsable, so it must not contain explanations or additional comments.""",

    "imperative": """A tag is a single word or short phrase describing a variable. Make tags general enough to apply to multiple variables and specific enough to separate similar variables. Tags will be used to identify causal directions between variables: make the set of tags per variable discriminative. Assign multiple tags per variable where appropriate.
Variables: $VARIABLES.

Generate the tag list now. Choose the tag count that balances expressivity and specificity. Never emit two tags covering the same set of variables. Output one line per tag: tag name, colon, comma-separated variables. Output nothing else. No explanations. No comments. No preamble. Machine-parsable output only.""",
}


def assemble_paraphrasing_templates():
    return {
        **{f"paraphrase_register_{k}": v for k, v in REGISTER.items()},
        **{f"paraphrase_politeness_{k}": v for k, v in POLITENESS.items()}}



from __future__ import annotations


BASE_WORLD_MODEL_PROMPT = """\
You are a world model for the {env_name} environment. Given the agent's current observation, its proposed action, and the task goal, predict what the agent will observe next.

Output format: a single paragraph starting with "Prediction:" describing the next observation in the style of the environment's own text output.
"""


EPISODIC_WORLD_MODEL_PROMPT = """\
You are a world model for the {env_name} environment. Given the agent's current observation, its proposed action, and the task goal, predict what the agent will observe next.

If a "## Retrieved similar past transitions" section is provided, use those transitions as analogies for what can change after this action. Let the retrieved transitions guide what changes, but keep the prediction consistent with the current observation and task goal.

Output format: a single paragraph starting with "Prediction:" describing the next observation in the style of the environment's own text output.
"""


SEMANTIC_WORLD_MODEL_PROMPT = """\
You are a world model for the {env_name} environment. Given the agent's current observation, its proposed action, and the task goal, predict what the agent will observe next.

If a "## Frame Axioms and Persistence Rules" section appears above, treat each rule as a constraint: do not predict any change to a property the rules say does not change for the action being taken. Use the rules to filter out spurious changes.

Output format: a single paragraph starting with "Prediction:" describing the next observation in the style of the environment's own text output.
"""


EPISODIC_SEMANTIC_WORLD_MODEL_PROMPT = """\
You are a world model for the {env_name} environment. Given the agent's current observation, its proposed action, and the task goal, predict what the agent will observe next.

If a "## Frame Axioms and Persistence Rules" section appears above, treat each rule as a constraint: do not predict any change to a property the rules say does not change for the action being taken.

If a "## Retrieved similar past transitions" section is provided, use those transitions as analogies for what can change after this action. Let the retrieved transitions guide what changes, and use the frame axioms / persistence rules to filter out spurious changes.

Output format: a single paragraph starting with "Prediction:" describing the next observation in the style of the environment's own text output.
"""


WORLD_MODEL_PROMPTS = {
    "base": BASE_WORLD_MODEL_PROMPT,
    "episodic": EPISODIC_WORLD_MODEL_PROMPT,
    "semantic": SEMANTIC_WORLD_MODEL_PROMPT,
    "episodic_semantic": EPISODIC_SEMANTIC_WORLD_MODEL_PROMPT,
}

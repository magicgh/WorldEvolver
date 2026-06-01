class BaseAgent:
    def __init__(self):
        super().__init__()

    def reset(self, goal, init_obs, init_act=None):
        pass

    def update(self, action, state):
        pass

    def run(self):
        pass

    @classmethod
    def from_config(cls, llm_model, config):
        pass

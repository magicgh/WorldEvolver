import abc

class BaseTask:
    def __int__(self):
        super().__int__()

        '''
        initialize llm and agent here for evaluation

        self.llm = load_llm(...)
        self.agent = load_agent(...)
        '''

    def evaluate_env(self, env_id):

        return

    def evaluate(self):

        return

    @classmethod
    def from_config(cls,
                    run_config,
                    llm_config,
                    agent_config,
                    env_config,
                    llm = None
                    ):

        return cls()

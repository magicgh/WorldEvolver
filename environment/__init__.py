from common.registry import registry


def load_environment(name, config):
    if name not in registry.list_environments():
        if name == "alfworld": from environment.alfworld.alfworld_env import AlfWorld
        if name == "scienceworld": from environment.scienceworld_env import Scienceworld

    env = registry.get_environment_class(name).from_config(config)
    return env

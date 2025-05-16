import yaml

def load_config(config_path: str = "E:\\2025\\Generative_AI\\Course\\Projects\\agentic-trading-bot\\config\\config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config
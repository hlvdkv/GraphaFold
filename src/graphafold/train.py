import click
import yaml

def load_config(config_path):
    """
    Load the configuration from a YAML file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        dict: Configuration parameters.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

@click.command()
@click.option('--config', default='config.yml', help='Path to the configuration file.')
def main(config):
    """
    Main function to start the training process.
    This function will read the configuration file and initiate the training process.
    """
    # Load configuration from the provided file
    config_params = load_config(config)
    print(f"Loaded configuration: {config_params}")

if __name__ == "__main__":
    main()
import click
import yaml
from torch.utils.data import DataLoader
from graphafold.data import TrainDataset, custom_collate

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
    train_ds = TrainDataset(config_params['data']['train_dir'])
    train_loader = DataLoader(train_ds, batch_size=config_params['batch_size'], shuffle=True, num_workers=config_params['num_workers'], collate_fn=custom_collate)
    print(f"Training dataset size: {len(train_ds)}")
    print("Batch sanity check:")
    for batch in train_loader:
        graphs, sequences, sequence_breaks = batch
        print(f"Graph batch size: {graphs.num_nodes()}")
        print(f"Graph edges batch size: {graphs.num_edges()}")
        print(f"Sequences batch size: {len(sequences)}")
        print(f"Sequence breaks batch size: {len(sequence_breaks)}")
        break

if __name__ == "__main__":
    main()
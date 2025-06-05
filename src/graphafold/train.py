import click
import yaml
from torch.utils.data import DataLoader
import lightning as L
import wandb
from graphafold.dataset import GraphDataset
from graphafold.data import custom_collate
from graphafold.model import GraphaFold

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
    L.seed_everything(42, workers=True)
    
    # Load configuration from the provided file
    config_params = load_config(config)
    print(f"Loaded configuration: {config_params}")
    train_ds = GraphDataset(config_params['data']['train_dir'])
    val_ds = GraphDataset(config_params['data']['val_dir'], validation=True)
    train_loader = DataLoader(train_ds, batch_size=config_params['batch_size'], shuffle=True, num_workers=config_params['num_workers'], collate_fn=custom_collate)
    val_loader = DataLoader(val_ds, batch_size=config_params['batch_size'], shuffle=False, num_workers=config_params['num_workers'], collate_fn=custom_collate)
    print(f"Training dataset size: {len(train_ds)}")
    print("Batch sanity check:")
    for batch in train_loader:
        graphs, sequences, edge_candidates, edge_labels = batch
        print(f"Graph batch size: {graphs.num_nodes()}")
        print(f"Graph edges batch size: {graphs.num_edges()}")
        print(f"Number of nodes per sample in batch: {graphs.batch_num_nodes()}")
        print(f"Sequences batch size: {len(sequences)}")
        print(f"Sample sequences: {sequences[:5]}")
        print(f"Edge candidates batch size: {edge_candidates.shape}")
        print(f"Edge labels batch size: {edge_labels.shape}")
        break

    if config_params['wandb']:
        wandb.init(project=config_params['wandb_project'], name=config_params['wandb_run_name'])
        wandb.config.update(config_params)
        logger = L.loggers.WandbLogger(
            project=config_params['wandb_project'],
            config=config_params
        )
    else:
        logger = None
    # Initialize the Lightning Trainer
    trainer = L.Trainer(
        max_epochs=config_params['epochs'],
        logger=logger,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(
                dirpath=config_params['checkpoint_dir'],
                filename='model-{epoch:02d}-{val_loss:.2f}',
                monitor='val_loss',
                save_top_k=1,
                mode='min'
            )
        ]
    )
    # Initialize the model
    model = GraphaFold(
        in_feats=config_params['model']['in_feats'],
        edge_feats=config_params['model']['edge_feats'],
        hidden_feats=config_params['model']['hidden_feats'],
        hidden_dim=config_params['model']['hidden_dim'],
        gcn_layers=config_params['model']['gcn_layers']
    )
    # Start training
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
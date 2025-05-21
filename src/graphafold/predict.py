import click
import dgl
import numpy as np
import torch
from torch.utils.data import DataLoader

from graphafold.utils import parse_dot2out
from graphafold.data import cmt2graph, GraphDataset, collate
from graphafold.model import GNNModel

def predict(idx_path: str, cmt_path: str) -> None:
    """
    Placeholder function for the prediction logic.
    This function should be implemented to perform the actual prediction.
    """
    graphs, edge_lists, labels_list, file_names = cmt2graph(cmt_path, idx_path)
    print("Edge lists:", edge_lists)
    dataset = GraphDataset(graphs, edge_lists, labels_list, file_names)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate)
    print(len(test_loader))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GNNModel(in_feats=4, edge_feats=2, hidden_feats=16, out_feats=2)
    model.load_state_dict(torch.load('checkpoints/model_v4.pth', map_location=device))
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for batched_graph, combined_edge_list, combined_labels, file_names_batch in test_loader:
            batched_graph = batched_graph.to(device)
            combined_edge_list = combined_edge_list.to(device)
            combined_labels = combined_labels.to(device)

            print(batched_graph)
            print(combined_edge_list)
            logits = model(batched_graph, combined_edge_list)
            _, predicted = torch.max(logits, 1)

            graphs_batch = dgl.unbatch(batched_graph)
            num_nodes_list = [g.number_of_nodes() for g in graphs_batch]
            cum_nodes = np.cumsum([0] + num_nodes_list)

            for i, graph in enumerate(graphs_batch):
                start = cum_nodes[i]
                end = cum_nodes[i+1]
                mask = (combined_edge_list[:, 0] >= start) & (combined_edge_list[:, 0] < end)
                edge_list = combined_edge_list[mask] - start
                predicted_labels = predicted[mask]

                predicted_edges = np.array([tuple(edge.cpu().numpy()) for edge, label in zip(edge_list, predicted_labels) if label == 1])
                print(f"Predicted edges for {file_names_batch[i]}: {predicted_edges}")



@click.command()
@click.option(
    "--input", "-i",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the input *.dot file",
)
@click.option(
    "--output", "-o",
    type=click.Path(dir_okay=True),
    default="./output",
    help="Path to the output dir, where the prediction results will be saved",
)
@click.option(
    "--idx", "-x",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the *.idx file",
)
@click.option(
    "--cmt", "-c",
    type=click.Path(exists=True, dir_okay=False),
    help="Path to the *.cmt file",
)
def main(input: str, output: str, idx, cmt) -> None:
    """
    Main function to run the GraphAFold prediction.
    """
    if not input and (not idx or not cmt):
        raise ValueError("Either input file or both idx and cmt files must be provided.")
    print(f"Input file: {input}")
    print(f"Output directory: {output}")
    if input:
        idx_path, cmt_path = parse_dot2out(input)
    elif idx and cmt:
        idx_path = idx
        cmt_path = cmt
    print(f"Index file: {idx_path}")
    print(f"CMT file: {cmt_path}")
    predict(idx_path, cmt_path)
    # TODO: load the input file
    # TODO: parse to graph
    # TODO: run the prediction
    # TODO: save the prediction results
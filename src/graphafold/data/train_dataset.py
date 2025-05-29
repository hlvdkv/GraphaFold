import os
import pickle
import dgl
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self, path):
        super(TrainDataset, self).__init__()
        self.path = path
        self.samples = os.listdir(path)
        self.graphs = []
        self.edge_lists = []
        self.labels_list = []
        self.file_names = []

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_path = os.path.join(self.path, sample)
        with open(sample_path, 'rb') as f:
            data = pickle.load(f)
        
        graph = dgl.graph(data.cn, num_nodes=data.num_nodes)
        graph.add_edges(data.neigboursp[:, 0], data.neighbours[:, 1]) # A-B


        return graph, data.sequences, data.sequnce_breaks
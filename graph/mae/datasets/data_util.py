from collections import Counter

import torch
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.utils import add_self_loops, remove_self_loops, degree

from torch_geometric.transforms import AddRandomWalkPE


def load_graph_classification_dataset(dataset_name, pe_dim=24, deg4feat=False):
    dataset_name = dataset_name.upper()
    dataset = TUDataset(root="./data", name=dataset_name,
                        pre_transform=AddRandomWalkPE(walk_length=pe_dim, attr_name='pe'))
    dataset = list(dataset)
    graph = dataset[0]

    if graph.x == None:
        if graph.y and not deg4feat and dataset_name != "REDDIT-BINARY":
            print("Use node label as node features")
            feature_dim = 0
            for g in dataset:
                feature_dim = max(feature_dim, int(g.y.max().item()))

            feature_dim += 1
            for i, g in enumerate(dataset):
                node_label = g.y.view(-1)
                feat = F.one_hot(
                    node_label, num_classes=int(feature_dim)).float()
                dataset[i].x = feat
        else:
            print("Using degree as node features")
            feature_dim = 0
            degrees = []
            for g in dataset:
                feature_dim = max(feature_dim, degree(
                    g.edge_index[0]).max().item())
                degrees.extend(degree(g.edge_index[0]).tolist())
            MAX_DEGREES = 400

            oversize = 0
            for d, n in Counter(degrees).items():
                if d > MAX_DEGREES:
                    oversize += n
            # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")
            feature_dim = min(feature_dim, MAX_DEGREES)

            feature_dim += 1
            for i, g in enumerate(dataset):
                degrees = degree(g.edge_index[0])
                degrees[degrees > MAX_DEGREES] = MAX_DEGREES
                degrees = torch.Tensor([int(x)
                                       for x in degrees.numpy().tolist()])
                feat = F.one_hot(degrees.to(torch.long),
                                 num_classes=int(feature_dim)).float()
                g.x = feat
                dataset[i] = g

    else:
        print("******** Use `attr` as node features ********")

    feature_dim = int(graph.num_features)

    labels = torch.tensor([x.y for x in dataset])

    num_classes = torch.max(labels).item() + 1
    for i, g in enumerate(dataset):
        dataset[i].edge_index = remove_self_loops(dataset[i].edge_index)[0]
        dataset[i].edge_index = add_self_loops(dataset[i].edge_index)[0]

    print(
        f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")
    return dataset, (feature_dim, num_classes)

import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WebKB
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected, degree
from torch_geometric.utils import scatter, k_hop_subgraph

import torch_geometric.transforms as T
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats


def load_dataset(dataset_name):
    if dataset_name in ("cora", "citeseer", "pubmed"):
        dataset = Planetoid("./data", dataset_name,
                            transform=T.NormalizeFeatures())
        graph = dataset[0]
        graph.edge_index = remove_self_loops(graph.edge_index)[0]
        graph.edge_index = add_self_loops(graph.edge_index)[0]
    else:
        if dataset_name in ("Photo"):
            dataset = Amazon(root='./data', name=dataset_name,
                             transform=T.NormalizeFeatures())
        if dataset_name in ("Physics", "CS"):
            dataset = Coauthor(root='./data', name=dataset_name,
                               transform=T.NormalizeFeatures())
        graph = dataset[0]

        graph.edge_index = to_undirected(graph.edge_index)
        graph.edge_index = remove_self_loops(graph.edge_index)[0]
        graph.edge_index = add_self_loops(graph.edge_index)[0]

        num_nodes = graph.num_nodes

        node_indices = list(range(num_nodes))

        test_idx, train_idx = train_test_split(
            node_indices, test_size=0.2, random_state=42)
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.5, random_state=42)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask[val_idx] = True
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[test_idx] = True

        graph.train_mask = train_mask
        graph.val_mask = val_mask
        graph.test_mask = test_mask

    num_features = dataset.num_features
    num_classes = dataset.num_classes
    return graph, (num_features, num_classes)


def extract_ego_graph(data, node_idx, num_hops):
    # Extract the subgraph for num_hops around the node_idx
    subgraph_nodes, subgraph_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx, num_hops, data.edge_index, relabel_nodes=True)

    # Extract the features and labels of nodes in the subgraph
    subgraph_node_features = data.x[subgraph_nodes] if data.x is not None else None
    subgraph_labels = data.y[subgraph_nodes] if data.y is not None else None

    mask_index = (subgraph_nodes == node_idx).nonzero().flatten()

    # Create a new Data object for the ego graph
    ego_data = Data(x=subgraph_node_features,
                    edge_index=subgraph_edge_index, y=subgraph_labels, mask_index=mask_index, subgraph_nodes=subgraph_nodes)

    return ego_data


def get_ego_graphs(name, num_hops: int = 1):
    if name in ("cora", "citeseer", "pubmed"):
        dataset = Planetoid(root='./data', name=name,
                            transform=T.NormalizeFeatures())
        graph = dataset[0]
        graph.edge_index = remove_self_loops(graph.edge_index)[0]
        graph.edge_index = add_self_loops(graph.edge_index)[0]
    else:
        if name in ("Photo"):
            dataset = Amazon(root='./data', name=name,
                             transform=T.NormalizeFeatures())
        if name in ("Physics", "CS"):
            dataset = Coauthor(root='./data', name=name,
                               transform=T.NormalizeFeatures())
        graph = dataset[0]

        graph.edge_index = to_undirected(graph.edge_index)
        graph.edge_index = remove_self_loops(graph.edge_index)[0]
        graph.edge_index = add_self_loops(graph.edge_index)[0]

        num_nodes = graph.num_nodes
        node_indices = list(range(num_nodes))
        test_idx, train_idx = train_test_split(
            node_indices, test_size=0.2, random_state=42)
        train_idx, val_idx = train_test_split(
            train_idx, test_size=0.5, random_state=42)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask[val_idx] = True
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask[test_idx] = True

        graph.train_mask = train_mask
        graph.val_mask = val_mask
        graph.test_mask = test_mask

    num_ego_graphs = graph.x.size(0)
    sub_graphs = []
    for i in range(num_ego_graphs):
        ego_graph = extract_ego_graph(graph, node_idx=i, num_hops=num_hops)
        sub_graphs.append(ego_graph)

    return sub_graphs, num_ego_graphs

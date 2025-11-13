from GNNLayer import GCNLayer
from torch import nn
import torch

class simpleGNN(nn.Module):
    def __init__(self, node_in, edge_in, hidden_dim, node_out, edge_out, global_out, num_gcn_layers=2):
        super(simpleGNN, self).__init__()

        self.gcn_layers = nn.ModuleList()
        for i in range(num_gcn_layers):
            n_in = node_in if i == 0 else node_out
            e_in = edge_in if i == 0 else edge_out
            self.gcn_layers.append(
                GCNLayer(n_in, e_in, edge_out, node_out)
            )
            
        # Additional layers for graph-level prediction (global regression/classification)
        # self.global_aggregation = nn.Sequential(
        #     nn.Linear(node_out, 1),  # Aggregating node features
        #     nn.ReLU()
        # )
        
        # self.global_prediction = nn.Sequential(
        #     nn.Linear(1, global_out),  # Final prediction layer (output)
        # )
        
        self.global_pool = 'sum'  # 'mean', 'sum', 'max', etc.
        
        self.readout = nn.Sequential(
            nn.Linear(node_out, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # For regression. Change output size if doing classification.
        )

    def forward(self, X, E, A):
        """
        Forward pass for the entire GNN model.
        Args:
            X: Node features [N, F_node]
            E: Edge features [N, N, F_edge]
            A: Adjacency matrix [N, N]
        Returns:
            Graph-level prediction (scalar per graph)
        """
        for gcn in self.gcn_layers:
            X, E = gcn(X, E, A)

        # Global graph-level prediction
        # graph_repr = self.global_aggregation(X)
        # out = self.global_prediction(graph_repr)
        
        # Global Pooling over nodes to get a graph-level representation
        if self.global_pool == 'mean':
            graph_repr = torch.mean(X, dim=0)
        elif self.global_pool == 'sum':
            graph_repr = torch.sum(X, dim=0)
        elif self.global_pool == 'max':
            graph_repr = torch.max(X, dim=0).values
        else:
            raise ValueError("Unknown global pooling method")
        out = self.readout(graph_repr)
        
        return out

# class simpleGNN(nn.Module):
    
#     def __init__(self, num_layers, num_heads, num_classes, dropout_rate):
#         self.num_layers = num_layers
#         self.num_heads = num_heads
#         self.num_classes = num_classes

#         for i in range(num_layers):
#             # Initialize GNN layers
#             self.add_module(f'layer_{i}', GNNLayer(num_heads, num_classes, dropout_rate))

#     def forward(self, x, edge_index):
#         # Implement the forward pass of the GNN here
#         pass

#     def train(self, data_loader):
#         # Implement the training loop here
#         pass

#     def evaluate(self, data_loader):
#         # Implement the evaluation loop here
#         pass

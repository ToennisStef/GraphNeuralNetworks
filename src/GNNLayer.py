import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, node_in, edge_in, edge_out, node_out, update_nodes=True, update_edges=True):
        """
        Graph Convolutional Layer (GCN) for updating node and edge features.
        """
        super(GCNLayer, self).__init__()
        self.update_nodes = update_nodes
        self.update_edges = update_edges

        self.edge_update = nn.Sequential(
            nn.Linear(edge_in + 2 * node_in, edge_out),
            nn.ReLU()
        )

        self.edge_to_node = nn.Sequential(
            nn.Linear(edge_out, edge_out),
            nn.ReLU()
        )
        
        self.node_update = nn.Sequential(
            nn.Linear(node_in + edge_out, node_out),
            nn.ReLU()
        )

    def forward(self, X, E, A):
        """
        Forward pass for the GCN layer.
        Args:
            X: Node features of shape (N, F_node)
            E: Edge features of shape (N, N, F_edge)
            A: Adjacency matrix of shape (N, N)
        Returns:
            Updated node features of shape (N, F_node_out)
            Updated edge features of shape (N, N, F_edge_out)
        """
        
        Xi, Xj = self.nodeToEdge_aggregation(X, A)
        E_input = torch.cat([E, Xi, Xj], dim=-1)
        E_updated = self.edge_update(E_input)
        E_updated = E_updated * A.unsqueeze(-1)  # Masking
        
        # Aggregate edge features into node space
        E_sum = E_updated.sum(dim=1)  # sum_j e_ij for each i
        E_msg = self.edge_to_node(E_sum)
        
        X_input = torch.cat([X, E_msg], dim=-1)
        X_updated = self.node_update(X_input)
        
        return X_updated, E_updated
        
    def node_pooling(self, X, A):
        """
        Node pooling layer that aggregates node features based on adjacency matrix A.
        Args:
            X: Node features of shape (N, F_node)
            A: Adjacency matrix of shape (N, N)
        Returns:
            Pooled node features of shape (N, F_node)
        """
        
        # Add self-loops
        I = torch.eye(A.size(0)).to(A.device)
        A_hat = A + I

        # Compute degree matrix
        D_hat = torch.diag(torch.sum(A_hat, dim=1))

        # Normalize A_hat
        D_hat_inv_sqrt = torch.linalg.inv(torch.sqrt(D_hat))
        A_norm = D_hat_inv_sqrt @ A_hat @ D_hat_inv_sqrt

        # Apply GCN update rule
        return A_norm @ X


    def edgeToNode_aggregation(self, E, A):
        """
        Aggregate edge features based on adjacency matrix A.
        Args:
            E: Edge features of shape (N, N, F_edge)
            A: Adjacency matrix of shape (N, N)
        Returns:
            Aggregated edge features of shape (N, F_edge)
        """
        
        # Mask by adjacency
        E_agg = (E * A.unsqueeze(-1)).sum(dim=0)    
        
        return E_agg

    def nodeToEdge_aggregation(self, X, A):
        """
        Aggregate node features based on adjacency matrix A.
        Args:
            X: Node features of shape (N, F_node)
            A: Adjacency matrix of shape (N, N)
        Returns:
            Aggregated node features of shape (N, F_node)
        """
        Xi = X.unsqueeze(1).expand(-1, A.size(0), -1)        # [N, N, F_node]
        Xj = X.unsqueeze(0).expand(A.size(0), -1, -1)        # [N, N, F_node]
        
        return Xi, Xj
    
class WeaveLayer(nn.Module):
    def __init__(self, node_in, edge_in, node_out, edge_out, update_nodes=True, update_edges=True):
        super().__init__()
        self.update_nodes = update_nodes
        self.update_edges = update_edges

        # MLPs to update node and edge features
        self.node_update = nn.Sequential(
            nn.Linear(node_in + edge_in, node_out),
            nn.ReLU()
        )

        self.edge_update = nn.Sequential(
            nn.Linear(edge_in + node_in, edge_out),
            nn.ReLU()
        )

    def forward(self, X, E, A):
        N = X.shape[0]

        if self.update_edges:
            # Get node pairs for each edge
            Xi = X.unsqueeze(1).expand(-1, N, -1)        # [N, N, F_node]
            Xj = X.unsqueeze(0).expand(N, -1, -1)        # [N, N, F_node]
            edge_input = torch.cat([E, Xi, Xj], dim=-1)
            # E = self.edge_update(edge_input) * A.unsqueeze(-1)
            E = self.edge_update(edge_input)             # [N, N, F_edge_out]
            E = E * A.unsqueeze(-1)                      # mask by adjacency


        if self.update_nodes:
            # Pool incoming edge features (sum over neighbors)
            E_agg = E.sum(dim=1)                         # [N, F_edge_out]
            node_input = torch.cat([X, E_agg], dim=-1)   # [N, F_node + F_edge_out]

            X = self.node_update(node_input)             # [N, F_node_out]
            
        return X, E

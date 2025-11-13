
class undirected_graph():
    """
    Class representation of an undirected graph.
    """
    def __init__(
        self, 
        global_vals:dict,
        vertex_vals:list, 
        edge_vals:list, 
        adj:set
        )->None:
        
        # pass input values
        self.global_vals:dict = global_vals
        self.vert_vals:list = vertex_vals
        self.edge_vals:list = edge_vals
        self.adj:set = adj
        
        # build vertex dict
        self.vertex_dict:dict = self._get_vertex_dict()
        
        # build adjacency dict
        self.adj_dict:dict = self._get_adjacency_dict(self.adj)
        
        # create a set of all verticies referenced in adj
        self.adj_verts:set = set().union(*adj)
        
        # build adjacency matrix
        self.adj_matrix:list = self._get_adjacency_matrix()
        
        # build edge matrix
        self.edge_matrix:list = self._get_edge_matrix()
        
        # set of connected verticies 
        self.connected_verts:set = self._get_connected_verticies(self.adj_dict)

        ## sanity checks for the graph
        # Check if the number of edge values matches the number of adjacencies
        assert len(self.edge_vals) == len(self.adj), "Number of edge values does not match the number of adjacencies"
        
        # check if the adjacency set is referencing a vertex/node that is NOT present  
        assert len(self.vert_vals) >= max(self.adj_verts), "Adjacency is referencing a non-existent vertex"
        
        # Warning if the graph consists of unconnected nodes
        # assert  len(self.connected_verts) != 1, "Graph consists of unconnected nodes or subgraphs"
        
        # ADD: warning if the graph consists of unconnected nodes!
        # ADD: create_subgraph() - creates a subgraph of the connected nodes and returns the graphs
        # ADD: plot() - plot the graph (color the verticies/nodes corresponding to the node )
        
        
    def _get_adjacency_dict(self, adj):
        """
        Create an adjacency dictionary from the adjacency set.
        """
        from collections import defaultdict, deque
        adj_dict =defaultdict(set)
        for edge in adj:
            u,v = tuple(edge)
            adj_dict[u].add(v)
            adj_dict[v].add(u)
        return adj_dict
    
    def _get_adjacency_matrix(self):
        """
        Create an adjacency matrix from the adjacency set.
        """
        n = len(self.vert_vals)
        adj_matrix = [[0 for _ in range(n)] for _ in range(n)]
        for edge in self.adj:
            u, v = tuple(edge)
            adj_matrix[u-1][v-1] = 1
            adj_matrix[v-1][u-1] = 1
        return adj_matrix

    def _get_edge_matrix(self):
        """
            Create an edge matrix from the adjacency set.
            """
        n = len(self.vert_vals)
        
        edge_features = len(self.edge_vals[0])
        edge_matrix = [[[0]*edge_features for _ in range(n)] for _ in range(n)]
        for i, edge in enumerate(self.adj):
            u, v = tuple(edge)
            edge_matrix[u-1][v-1] = self.edge_vals[i]
            edge_matrix[v-1][u-1] = self.edge_vals[i]
        return edge_matrix
    
    def _get_vertex_dict(self):
        """
        Create a dictionary of vertex values.
        """
        vert_dict = {}
        for i, value in enumerate(self.vert_vals):
            vert_dict[i+1] = value
        return vert_dict

    def _get_edge_dict(self):
        """
        Create a dictionary of edge values with keys as elements of adj.
        """
        edge_dict = {}
        for edge, value in zip(self.adj, self.edge_vals):
            edge_dict[tuple(edge)] = value
        return edge_dict
    
    def _get_connected_verticies(self, adj_dict):
        from collections import deque
        
        # Traverse graph to find connected components
        visited = set()
        components = set()
        
        for node in adj_dict:
            if node not in visited:
                component = set()
                queue = deque([node])
                
                while queue:
                    current = queue.popleft()
                    if current not in visited:
                        visited.add(current)
                        component.add(current)
                        queue.extend(adj_dict[current]-visited)
                components.add(frozenset(component))
        return components
    
    def plot(self, output_dir="graph_viz", open_browser=True, port=8000):
        """
        Generate an interactive D3 plot of the graph.
        """
        import json
        import webbrowser
        from pathlib import Path
        import os
        
        # Step 1: Build data structure for D3
        nodes = [{"id": i} for i in range(1, len(self.vert_vals) + 1)]
        links = [{"source": u, "target": v} for u, v in self.adj]
        # Add vertex and edge values to the graph data
        for i, node in enumerate(nodes):
            node["value"] = self.vert_vals[i]

        for i, link in enumerate(links):
            link["value"] = self.edge_vals[i]
        
        # Add global values to the graph data
        graph_data = {
            "global_vals": self.global_vals,
            "nodes": nodes,
            "links": links,
        }
        
        # Step 2: Write JSON
        output_dir = (Path(__file__).parent / output_dir).resolve()
        output_dir.mkdir(exist_ok=True)
        json_path = output_dir / "graphData.json"
        with open(json_path, "w") as f:
            json.dump(graph_data, f)
        html_path = Path(output_dir) / "graph_viz.html"

        absolute_html_path = html_path.resolve()
        

class molecule_graph(undirected_graph):
    """
    Class representation of a molecule graph.
    Inherits from the undirected_graph class.
    """
    def __init__(
        self, 
        global_vals:list, 
        vertex_vals:list, 
        edge_vals:list, 
        adj:set,
        SMILES:str,
        COSMO_name:str,
        atoms:list,
        bonds:list
        ):
        """
        Initialize the molecule graph with global, vertex, edge values and adjacency set.
        """
        super().__init__(global_vals, vertex_vals, edge_vals, adj)
        self.SMILES = SMILES
        self.COSMO_name = COSMO_name
        self.atoms = atoms
        self.bonds = bonds
        
    def plot(self, output_dir="molecule_viz", open_browser=True, port=8000):
            """
            Generate an interactive D3 plot of the graph.
            """
            import json
            import webbrowser
            from pathlib import Path
            import os
            
            # Step 1: Build data structure for D3
            nodes = [{"id": i} for i in range(1, len(self.vert_vals) + 1)]
            links = [{"source": u, "target": v} for u, v in self.adj]
            # Add vertex and edge values to the graph data
            for i, node in enumerate(nodes):
                node["value"] = self.vert_vals[i]
                node["atom"] = self.atoms[i]

            for i, link in enumerate(links):
                link["value"] = self.edge_vals[i]
                link["bond"] = self.bonds[i]
            
            # Add global values to the graph data
            graph_data = {
                "global_vals": self.global_vals,
                "nodes": nodes,
                "links": links,
            }
            
            # Step 2: Write JSON
            output_dir = (Path(__file__).parent / output_dir).resolve()
            output_dir.mkdir(exist_ok=True)
            json_path = output_dir / "graphData.json"
            with open(json_path, "w") as f:
                json.dump(graph_data, f)
            html_path = Path(output_dir) / "graph_viz.html"

            absolute_html_path = html_path.resolve()

    
    # # Add additional element to the graphData.json
    # graph_data["additional_info"] = {
    #     "SMILES": self.SMILES,
    #     "COSMO_name": self.COSMO_name,
    #     "atoms": self.atoms,
    #     "bonds": self.bonds
    # }
        
        
        # Ensure the path is correct in WSL and Windows environments
        # if open_browser:
        #     if os.name == "nt":  # On Windows, open the HTML file using the default browser
        #         webbrowser.open(f"file:///{absolute_html_path}")
        #     else:  # On WSL, ensure you use the right file path format for Linux
        #         # This opens the HTML in the default browser by referencing it with 'file://'
                
        #         # os.system(f"xdg-open {absolute_html_path}")
        #         # webbrowser.open(f"file://{html_path}")
        #         webbrowser.open(f"file://{absolute_html_path}")
        
        
    # def _make_graph_json(self):
    #     """
    #     Create a JSON representation of the graph.
    #     Use this to create a graph in the browser via html and js(d3.js).
    #     """
        
        
    #     raise NotImplementedError("Graph JSON creation is not implemented yet.")
        
    # def plot(self):
    #     import plotly.graph_objects as go
    #     import random
        
    #     # pos = {v: (random.uniform(-1,1), random.uniform(-1,1)) for v in self.adj_verts}
    #     pos = self.spring_layout(
    #         verts=self.adj_verts, 
    #         edges=self.adj,
    #         iteration=50, 
    #         k=None, 
    #         width=1, 
    #         height=1
    #     )
        
    #     # Node traces
    #     node_x = []
    #     node_y = []
    #     node_text = []
    #     for v in self.adj_verts:
    #         x, y = pos[v]
    #         node_x.append(x)
    #         node_y.append(y)
    #         # label = f"Node {v}"
    #         node_text.append(f"Node {v}: {self.vert_vals[v-1]}")
        
    #     node_trace = go.Scatter(
    #         x=node_x,
    #         y=node_y,
    #         mode='markers+text',
    #         text=node_text,
    #         textposition="top center",
    #         marker=dict(size=10, color='blue', line=dict(width=2)),
    #         hoverinfo='text',
    #         hovertext=node_text
    #     )
        
    #     # Edge traces
    #     edge_x = []
    #     edge_y = []
    #     edge_text = []
    #     for edge in self.adj:
    #         u,v = tuple(edge)
    #         x0, y0 = pos[u]
    #         x1, y1 = pos[v]
    #         edge_x += [x0, x1, None]
    #         edge_y += [y0, y1, None]

    #         edge_text.append(f"Edge {u}-{v}: something here")

    #     edge_trace = go.Scatter(
    #         x=edge_x,
    #         y=edge_y,
    #         line=dict(width=1, color='black'),
    #         hoverinfo='text',
    #         mode='lines',
    #         text=edge_text,
    #         hovertext=edge_text
    #     )
    #     # Create figure
    #     fig = go.Figure(data=[edge_trace, node_trace],
    #                     layout=go.Layout(
    #                         title="Undirected Graph",
    #                         # titlefont_size=16,
    #                         showlegend=False,
    #                         hovermode='closest',
    #                         margin=dict(b=0,l=0,r=0,t=0),
    #                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    #                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    #                     ))
    #     fig.show()
        
    # def spring_layout(
    #     self, 
    #     verts, 
    #     edges, 
    #     iteration=1500, 
    #     k=None, 
    #     width=1, 
    #     height=1,
    #     ):
        
    #     import random
    #     import math
    #     pos = {v: (random.uniform(-0.5,0.5), random.uniform(-0.5,0.5)) for v in verts}
        
    #     max_disp = 0.1
    #     if k is None:
    #         k = math.sqrt(width*height/len(verts))
        
    #     def distance(p1,p2):
    #         return max(math.dist(p1,p2) + 0.0001, 0.01)
        
    #     for _ in range(iteration):
    #         disp = {v: [0,0] for v in verts}
            
    #         # Repulsion forces
    #         for v in verts:
    #             for u in verts:
    #                 if u == v:
    #                     continue
    #                 dx, dy = pos[u][0]-pos[v][0], pos[u][1]-pos[v][1]
    #                 dist = distance(pos[u], pos[v])
    #                 force = k**2/dist*1.25
    #                 disp[v][0] += force * dx/dist
    #                 disp[v][1] += force * dy/dist
                    
    #         # Attraction forces
    #         for edge in edges:
    #             u,v = tuple(edge)
    #             dx, dy = pos[v][0]-pos[u][0], pos[v][1]-pos[u][1]
    #             dist = distance(pos[v], pos[u])
    #             force = dist**2/k 
    #             delta = [(dx/dist)*force, (dy/dist)*force]
    #             disp[u][0] -= delta[0]
    #             disp[u][1] -= delta[1]
    #             disp[v][0] += delta[0]
    #             disp[v][1] += delta[1]

    #         # update positions
    #         for v in verts:
    #             dx, dy = disp[v]
                
    #             mag = math.sqrt(disp[v][0]**2 + disp[v][1]**2)
    #             if mag > max_disp:
    #                 dx, dy = (dx/mag)*max_disp, (dy/mag)*max_disp
    #             pos[v] = (pos[v][0] + dx, pos[v][1] + dy)
                
        
    #     return pos
        
        
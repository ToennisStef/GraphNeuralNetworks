from Graphs import undirected_graph

# hexane
hexane_global_vals = [0]

hexane_vertex_vals = [[1],[1],[1],[1],[1],[1]]
# 1 - carbon, 2 - Oxygen, 3 

hexane_edge_vals = [[1],[1],[1],[1],[1]]
# edge values represent atom bond type (1 - covalent bond, add correct bond types here)


hexane_adjacency = {
    frozenset([1,2]),
    frozenset([2,3]),
    frozenset([3,4]),
    frozenset([4,5]),
    frozenset([5,6])
}
# haxane connectivity (undirected graph)

hexane = undirected_graph(
    global_vals=hexane_global_vals,
    vertex_vals=hexane_vertex_vals,
    edge_vals=hexane_edge_vals,
    adj=hexane_adjacency
    )

# haxane
# sanety_check_1 = len(haxane.vert_vals)>=max(haxane.connected_verts)
print(f"adj_dict: {hexane.adj_dict}")
print(f"len(haxane.vertex): {len(hexane.vert_vals)}")
print(f"connected_verts: {hexane.connected_verts}")
# print(f"max(haxane.all_verticies): {max(haxane.connected_verts)}")
# print(f"sanety_check_1: {sanety_check_1}")


# randomgraph
# adjacency = {
#     frozenset([1,2]),
#     frozenset([2,3]),
#     frozenset([3,4]),
#     frozenset([4,5])
# }

hexane.plot()
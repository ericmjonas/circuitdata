import networkx as nx
import numpy as np

def remove_merge_wires(g, ntype='wire'):
    """
    for all nodes that are 'wires', remove the node and connect
    all of their neighborhood directly to one another. 
    """
    # get the list of wire nodes
    wires = [n for n in g.nodes() if g.node[n]['ntype'] == 'wire']
    
    for node_w in wires:
        neighbors = g.neighbors(node_w)
        g.remove_node(node_w)
        for n1 in neighbors:
            for n2 in neighbors:
                if n1 != n2:
                    g.add_edge(n1, n2)


    
def get_conn_matrix(g, tfdf, src_pin, dest_pin):
    """
    Create the adj matrix for a particular set of pins
    
    """
    canonical_node_ordering = tfdf.index.values
    node_pos_lut = {n : p for p, n in enumerate(canonical_node_ordering)}
    
    N = len(canonical_node_ordering)
    adj_mat = np.zeros((N, N), dtype = np.uint8)
    
    for tgt_node in canonical_node_ordering:
        conn_list = []
        for n in g.neighbors(tgt_node):

            if g.edge[tgt_node][n]['pin'] == src_pin:
                # this is the node that is connected to this pin
                for other_n in g.neighbors(n):
                    if other_n == tgt_node and g[n][other_n]['pin'] == src_pin:
                        pass #don't connect to ourselves
                    else:
                        if g[n][other_n]['pin'] == dest_pin:
                            conn_list.append(other_n)
        for t in conn_list:
            adj_mat[node_pos_lut[tgt_node], node_pos_lut[t]] = 1
    return adj_mat

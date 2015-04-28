import simplejson as json
import pandas
import os
import networkx as nx
import numpy as np
from matplotlib import pylab
from ruffus import * 
import cPickle as pickle
import matplotlib
import util

DATA_DIR = "../../data/netlists"
@files(os.path.join(DATA_DIR, "data.json"), 'data.pickle')
def load_data(input_file, output_file):
    data = json.load(open(input_file, 'r'))
    nodenames = data['nodenames']
    nodei_to_name = {v:k for k, v in nodenames.items()}
    transdefs = data['transdefs']
    segdefs = data['segdefs']

    # What they call nodes we really call edges
    nodes = {}

    for seg in segdefs:
        w = seg[0]
        if w not in nodes:
            nodes[w] = {'segs': [], 
                        'pullup' : seg[1] == '+', 
                        'gates' : [],
                        'c1c2s' : []}
        nodes[w]['segs'].append(seg[3:])

    # trans are the 
    transistors = {}
    for tdef in transdefs:
        name = tdef[0]
        gate = tdef[1]
        c1 = tdef[2]
        c2 = tdef[3]
        bb = tdef[4]
        trans = {'name' : name, 
                 'on' : False, 
                 'gate' : gate, 
                 'c1' : c1, 
                 'c2' : c2, 
                 'bb' : bb}


        nodes[gate]['gates'].append(name)
        nodes[c1]['c1c2s'].append(name)
        nodes[c2]['c1c2s'].append(name)
        transistors[name] = trans

    # sort nodes by gate count

    df = pandas.DataFrame({'pullup' : [n['pullup'] for n in nodes.values()],
                           'gates' : [len(n['gates']) for n in nodes.values()], 
                           'c1c2s' : [len(n['c1c2s']) for n in nodes.values()]}, 
                          index=nodes.keys())
    df['name'] = pandas.Series(nodei_to_name)

    print "nodes sorted by gates" 
    
    result = df.sort(['gates'], ascending=False)
    print result.head(10)

    print "nodes sorted by c1c2s"
    result = df.sort(['c1c2s'], ascending=False)
    print result.head(10)

    tfdf = pandas.DataFrame(transistors.values(), index=transistors.keys())

    #tfdf['x'] = Series([tfdf['bb'][0] - tfdf['bb'][1])/2. 
    tfdf['x'] =  tfdf['bb'].map(lambda x : (x[0] + x[1])/2.0)
    tfdf['y'] =  tfdf['bb'].map(lambda x : (x[2] + x[3])/2.0)

    # 
    pickle.dump({'tfdf' : tfdf, 'wiredf'  : df}, 
                open(output_file, 'w'))

# x1 y1 x2 y2
 
REGIONS = {'decode' : {'x': (750, 8400), 'y': (7600, 8800)}, 
           'xysregs' : {'x' : (1400, 3000), 'y' : ( 1000, 4300)}, 
           'lower' : {'x' : (1350, 7700), 'y' : ( 1000, 4350)}, 
           'all' : {'x' : (0, 9000), 'y' : (0, 10000)}}



@files(load_data, "transistors.pdf")
def plot_transistors(input_file, output_file):
    d = pickle.load(open(input_file, 'r'))
    tfdf = d['tfdf']

    f = pylab.figure()
    ax = f.add_subplot(1, 1, 1)
    ax.scatter(tfdf['x'], tfdf['y'], s=5, edgecolor='none', c='k', alpha=0.5)
    ax.set_xlim(0, 9000)
    ax.set_ylim(0, 10000)
    ax.set_xlabel("um")
    ax.set_ylabel("um")

    ax.set_aspect(1.0)
    f.tight_layout()
    f.savefig(output_file, bbox_inches='tight')

@files(load_data, "transistors.regions.pdf")
def plot_transistors_regions(input_file, output_file):
    d = pickle.load(open(input_file, 'r'))
    tfdf = d['tfdf']

    f = pylab.figure()
    ax = f.add_subplot(1, 1, 1)
    ax.scatter(tfdf['x'], tfdf['y'], s=5, edgecolor='none', c='k', alpha=0.5)

    for region_name, r in REGIONS.iteritems():
        ax.add_patch(matplotlib.patches.Rectangle((r['x'][0], r['y'][0]), 
                                                  r['x'][1] - r['x'][0], 
                                                  r['y'][1] - r['y'][0], 
                                                  facecolor='none'))
        
    ax.set_xlim(0, 9000)
    ax.set_ylim(0, 10000)
    ax.set_xlabel("um")
    ax.set_ylabel("um")

    ax.set_aspect(1.0)
    f.tight_layout()
    f.savefig(output_file, bbox_inches='tight')

@files(load_data, "graph.pickle")
def create_raw_graph(input_file, output_file):
    d = pickle.load(open(input_file, 'r'))
    tfdf = d['tfdf']
    wiredf = d['wiredf']
    g=nx.Graph()
    g.add_nodes_from(wiredf.index, ntype='wire')
    g.add_nodes_from(tfdf['name'], ntype='transistor')

    for rowi, row in tfdf.iterrows():
        t_node = row['name']
        for pin in ['gate', 'c1', 'c2']:
            g.add_edge(t_node, row[pin], pin =pin)

    pickle.dump({'graph' : g}, 
                open(output_file, 'w'))

@files([load_data, create_raw_graph], 'rawgraph.dot')
def plot_raw_graph((data_file, graph_file), output_file):

    d = pickle.load(open(data_file, 'r'))
    tfdf = d['tfdf']
    wiredf = d['wiredf']
    
    gf = pickle.load(open(graph_file, 'r'))
    g = gf['graph']
    
    nodes_to_delete = ['vss', 'vcc']
    for n in nodes_to_delete:
        r = wiredf[wiredf['name'] == n].iloc[0]
        g.remove_node(r.name)
        
    for n in g.nodes():
        if g.node[n]['ntype'] == 'wire':
            g.node[n]['fillcolor'] = '#FF000080'
            wire_name = wiredf.loc[n]['name']
            if isinstance(wire_name, float):
                wire_name = ""
            g.node[n]['label'] = wire_name
        elif g.node[n]['ntype'] == 'transistor':
            g.node[n]['fillcolor'] = '#0000FF80'
        g.node[n]['style'] = 'filled'
    nx.write_dot(g, output_file)

    #sfdp -Tpdf test.dot -o test.pdf -v -Goverlap=prism

@files([load_data, create_raw_graph], 'graph.merged.pickle')
def merge_wires_graph((data_file, graph_file), output_file):
    """
    Merge the wires into the transistor nodes, 
    filter out vcc and vss

    """

    d = pickle.load(open(data_file, 'r'))
    tfdf = d['tfdf']
    wiredf = d['wiredf']
    
    gf = pickle.load(open(graph_file, 'r'))
    g = gf['graph']
    
    nodes_to_delete = ['vss', 'vcc']
    for n in nodes_to_delete:
        r = wiredf[wiredf['name'] == n].iloc[0]
        g.remove_node(r.name)

    util.remove_merge_wires(g)

    pickle.dump({"graph" : g}, 
                open(output_file, 'w'))

    #sfdp -Tpdf test.dot -o test.pdf -v -Goverlap=prism

def get_nodes_non_gate(graph, wire_node):
    # for a wire node, get all of the connecting nodes that are connected by non-gates
    nlist = []
    for n in graph.neighbors(wire_node):
       if graph[wire_node][n]['pin'] != 'gate':
            nlist.append(n)
    return nlist

@files([load_data, create_raw_graph], 'graph.dir.pickle')
def create_dir_graph((data_file, graph_file), output_file):
    d = pickle.load(open(data_file, 'r'))
    tfdf = d['tfdf']
    wiredf = d['wiredf']
    
    gf = pickle.load(open(graph_file, 'r'))
    graph = gf['graph']
        
    G = nx.DiGraph()

    for node in graph.nodes_iter():
        if graph.node[node]['ntype'] == 'transistor':
            G.add_node(node)

    for node in graph.nodes_iter():
        if graph.node[node]['ntype'] == 'transistor': 
           for edge in graph.edges(node):
                if graph[edge[0]][edge[1]]['pin'] == 'gate':
                    # this is my gate, so find the voltage node and get the drivers
                    for driver_trans in get_nodes_non_gate(graph, edge[1]):
                        G.add_edge(driver_trans, node)
    pickle.dump({'graph' : G}, 
                open(output_file, 'w'))

@files(merge_wires_graph, 'mergedgraph.dot')
def plot_merged_graph(graph_file, output_file):

    gf = pickle.load(open(graph_file, 'r'))
    g = gf['graph']
    nx.write_dot(g, output_file)

    #sfdp -Tpdf test.dot -o test.pdf -v -Goverlap=prism

@files([load_data, create_raw_graph],"analysis.txt")
def analysis((data_file, graph_file), out_file):

    d = pickle.load(open(data_file, 'r'))
    tfdf = d['tfdf']
    wiredf = d['wiredf']

    gf = pickle.load(open(graph_file, 'r'))
    g = gf['graph']
        
    for n in g.nodes():
        if g.node[n]['ntype'] == 'wire':
            g.node[n]['fillcolor'] = '#FF000080'
            wire_name = wiredf.loc[n]['name']
            if isinstance(wire_name, float):
                wire_name = ""
            g.node[n]['label'] = wire_name
        elif g.node[n]['ntype'] == 'transistor':
            g.node[n]['fillcolor'] = '#0000FF80'
            g.node[n]['label'] = str(n)
        else:
            raise NotImplementedError()
        g.node[n]['style'] = 'filled'
    

    vcc = wiredf[wiredf['name'] == 'vcc'].iloc[0].name
    vss = wiredf[wiredf['name'] == 'vss'].iloc[0].name

    # how many nodes have vcc on a c1 or c2
    print "c1 -> vcc: ", len(tfdf[tfdf['c1'] == vcc])
    print "c2 -> vcc: ", len(tfdf[tfdf['c2'] == vcc])
    print "gate -> vcc: ", len(tfdf[tfdf['gate'] == vcc])
    
    print "c1 -> vss: ", len(tfdf[tfdf['c1'] == vss])
    print "c2 -> vss: ", len(tfdf[tfdf['c2'] == vss])
    print "gate -> vss: ", len(tfdf[tfdf['gate'] == vss])
    
    IGNORE_WIRES = set([vcc, vss])
    # get the neighbors 
    tgt_trans = ['t%03d' % (i*100) for i in range(1, 30)]

    for t in tgt_trans:
        active_set = set([t])
        ITERS = 3
        for i in range(ITERS):
            new_set = set()
            for n in active_set:
                new_set.update(set(g.neighbors(n)))
            active_set = active_set.union(new_set)

            active_set.difference_update(IGNORE_WIRES)
        active_set.update(IGNORE_WIRES)
        print len(active_set)
        sg = g.subgraph(active_set).copy()

        f = pylab.figure(figsize=(16, 16))
        ax = f.add_subplot(1,1, 1)
        labels = {k : g.node[k]['label'] for k in sg.nodes()}


        
        for pnode in IGNORE_WIRES:
            conn_to_pnode = sg.neighbors(pnode)
            pnode_name = sg.node[pnode]['label']
            for n_i, n in enumerate(conn_to_pnode):
                pin_name = sg.edge[pnode][n]['pin']
                sg.remove_edge(pnode, n)
                new_pnode_name = "%s.%d" % (pnode_name, n_i)
                sg.add_node(new_pnode_name, ntype='wire', power=True)
                sg.add_edge(n, new_pnode_name, pin=pin_name)
                labels[new_pnode_name] = pnode_name
            sg.remove_node(pnode)
            del labels[pnode]

        edge_labels = {}
        for k1, k2 in sg.edges():
            edge_labels[(k1, k2)] = sg.edge[k1][k2]['pin'] 

        node_sizes = []
        node_colors = []
        for n in sg.nodes():
            s = 1000
            c = 'r'
            if sg.node[n]['ntype'] == 'wire':
                s = 800
                c = 'w'
                if 'power' in sg.node[n]:
                    c = 'k'
                    s = 100
            if n == t:
                c = 'b'
                s = 1000
            node_sizes.append(s)
            node_colors.append(c)

        pos = nx.spring_layout(sg, scale=4.0)

        nx.draw_networkx_nodes(sg, pos, ax=ax,
                               labels = labels, 
                               node_size = node_sizes,
                               node_color = node_colors)

        nx.draw_networkx_edges(sg, pos, ax=ax)
        nx.draw_networkx_labels(sg, pos, ax=ax,
                                labels=labels, font_size=6)
        nx.draw_networkx_edge_labels(sg, pos, ax=ax,
                                     edge_labels=edge_labels, 
                                     font_size=6)

        f.savefig("graph.%s.pdf" % t)

# @files([load_data, create_raw_graph], 'multiple_relations.pickle')
# def merge_wires_graph((data_file, graph_file), output_file):
#     """
#     Create a collection of 6 symmetric adj matrices
#     G <-> G
#     C1 <-> C1
#     C2 <-> C2
#     G -> C1
#     G -> C2
#     C1 -> C2
    
#     """

#     d = pickle.load(open(data_file, 'r'))
#     tfdf = d['tfdf']
#     wiredf = d['wiredf']
    
#     gf = pickle.load(open(graph_file, 'r'))
#     g = gf['graph']
    
#     nodes_to_delete = ['vss', 'vcc']
#     for n in nodes_to_delete:
#         r = wiredf[wiredf['name'] == n].iloc[0]
#         g.remove_node(r.name)

#     util.remove_merge_wires(g)

#     pickle.dump({"graph" : g}, 
#                 open(output_file, 'w'))
    
@files([load_data, merge_wires_graph], 'all.adjmat.pickle')
def create_all_adj_matrix((data_file, graph_file), output_file):
    """
    Take the merged graph and compute the adjacency matrix! 
    
    """

    d = pickle.load(open(data_file, 'r'))
    tfdf = d['tfdf']
    wiredf = d['wiredf']
    
    gf = pickle.load(open(graph_file, 'r'))
    g = gf['graph']

    canonical_node_ordering = tfdf.index
    N = len(canonical_node_ordering)
    adj_mat = np.zeros((N, N), dtype = [('link', np.uint8), 
                                        ('distance', np.float32)])
                  
    print "now walk"
    # create graph
    for n1_i, (n1, n1_data) in enumerate(tfdf.iterrows()):
        x1 = n1_data['x']
        y1 = n1_data['y']
        print n1_i
        for n2_i, (n2, row_data) in enumerate(tfdf.iterrows()):
            if g.has_edge(n1, n2):
                adj_mat[n1_i, n2_i]['link'] =True
            x2 = row_data['x']
            y2 = row_data['y']
            d = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            adj_mat[n1_i, n2_i]['distance'] = d
    pickle.dump({'adj_mat' : adj_mat}, 
                open(output_file, 'w'))

@files([load_data, create_raw_graph], 'typed.adjmat.pickle')
def create_typed_adj_matrix((data_file, graph_file), output_file):
    """
    
    """

    d = pickle.load(open(data_file, 'r'))
    tfdf = d['tfdf']
    wiredf = d['wiredf']
    
    gf = pickle.load(open(graph_file, 'r'))
    g = gf['graph']

    adj_mats = {}
    dtype = [('distance', np.float32)]
    PIN_PAIRS = [('gate', 'gate'), ('gate', 'c1'), ('gate', 'c2'), 
                 ('c1', 'c1'), ('c1', 'c2'), ('c2', 'c2')]
    for s, d in PIN_PAIRS:
        
        adj_mats[(s, d)] = util.get_conn_matrix(g, tfdf, s, d)
        dtype.append(('%s_%s' % (s, d), np.uint8))

    canonical_node_ordering = tfdf.index
    N = len(canonical_node_ordering)
    dist_mat = np.zeros((N, N), dtype =np.float32)
                  

    for n1_i, (n1, n1_data) in enumerate(tfdf.iterrows()):
        x1 = n1_data['x']
        y1 = n1_data['y']
        for n2_i, (n2, row_data) in enumerate(tfdf.iterrows()):
            x2 = row_data['x']
            y2 = row_data['y']
            d = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            dist_mat[n1_i, n2_i] = d

    adj_mat = np.zeros((N, N), dtype=dtype)
    adj_mat['distance'] = dist_mat
    for s, d in PIN_PAIRS:
        adj_mat['%s_%s' % (s, d)] = adj_mats[(s, d)]

    pickle.dump({'adj_mats' : adj_mats, 
                 'dist_mat' : dist_mat, 
                 'adj_mat' : adj_mat, 
                 'pin_pairs' : PIN_PAIRS},
                open(output_file, 'w'))

@files([load_data, merge_wires_graph], 'count.adjmat.pickle')
def create_count_adj_matrix((data_file, graph_file), output_file):
    """
    Take the merged graph and compute the adjacency matrix 
    of count data
    
    """

    d = pickle.load(open(data_file, 'r'))
    tfdf = d['tfdf']
    wiredf = d['wiredf']
    
    gf = pickle.load(open(graph_file, 'r'))
    g = gf['graph']

    canonical_node_ordering = tfdf.index
    N = len(canonical_node_ordering)
    adj_mat = np.zeros((N, N), dtype = [('link', np.int32), 
                                        ('distance', np.float32)])
                  
    print "now walk"
    # create graph
    for n1_i, (n1, n1_data) in enumerate(tfdf.iterrows()):
        x1 = n1_data['x']
        y1 = n1_data['y']
        print n1_i
        for n2_i, (n2, row_data) in enumerate(tfdf.iterrows()):
            if g.has_edge(n1, n2):
                adj_mat[n1_i, n2_i]['link']  += 1
            x2 = row_data['x']
            y2 = row_data['y']
            d = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            adj_mat[n1_i, n2_i]['distance'] = d
    pickle.dump({'adj_mat' : adj_mat}, 
                open(output_file, 'w'))


@files([load_data, create_dir_graph], 'dir.adjmat.pickle')
def create_dir_adj_matrix((data_file, graph_file), output_file):
    """
    Take the merged graph and compute the adjacency matrix! 
    
    """

    d = pickle.load(open(data_file, 'r'))
    tfdf = d['tfdf']
    wiredf = d['wiredf']
    
    gf = pickle.load(open(graph_file, 'r'))
    g = gf['graph']

    canonical_node_ordering = tfdf.index
    N = len(canonical_node_ordering)
    adj_mat = np.zeros((N, N), dtype = [('link', np.uint8), 
                                        ('distance', np.float32)])
                  
    print "now walk"
    # create graph
    for n1_i, (n1, n1_data) in enumerate(tfdf.iterrows()):
        x1 = n1_data['x']
        y1 = n1_data['y']
        print n1_i
        for n2_i, (n2, row_data) in enumerate(tfdf.iterrows()):
            if g.has_edge(n1, n2):
                adj_mat[n1_i, n2_i]['link'] =True
            x2 = row_data['x']
            y2 = row_data['y']
            d = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            adj_mat[n1_i, n2_i]['distance'] = d
    pickle.dump({'adj_mat' : adj_mat}, 
                open(output_file, 'w'))

@transform([create_all_adj_matrix, create_dir_adj_matrix], 
           suffix(".adjmat.pickle"), [".adj.png", ".distances.png"])
def plot_adj_matrix(infile, (adj_matrix_outfile, distance_outfile)):
    d = pickle.load(open(infile, 'r'))
    f = pylab.figure(figsize=(12, 12))
    ax_link = f.add_subplot(1, 1, 1)
    adj_mat = d['adj_mat']
    #ax_dist = f.add_subplot(1, 2, 2)
    
    links = np.argwhere(adj_mat['link'])
    print links.shape
    ax_link.imshow(adj_mat['link'], interpolation='nearest', cmap = pylab.cm.Greys)
    #ax_link.scatter(links[:, 0], links[:, 1], edgecolor='none', alpha=0.5, 
    #                s=1)

    # ax_dist.imshow(adj_mat['distance'], 
    #                interpolation='nearest')
    ax_link.set_xlim(0, len(adj_mat))
    ax_link.set_ylim(len(adj_mat), 0)
    f.tight_layout()
    f.savefig(adj_matrix_outfile, dpi=300)
    print "next fig" 
    f = pylab.figure(figsize=(12, 12))
    ax_alldist = f.add_subplot(2, 1, 1)
    ax_alldist.hist(adj_mat['distance'].flat, bins=40)
    
    adj_dist_flat = adj_mat['distance'].flatten()
    idx = np.nonzero(adj_mat['link'].flat)
    conn_only = adj_dist_flat[idx]
    
    ax_conndist = f.add_subplot(2, 1, 2)
    ax_conndist.hist(conn_only.flat, bins=40)
    f.savefig(distance_outfile)

@transform([create_all_adj_matrix, create_dir_adj_matrix, 
            create_count_adj_matrix, create_typed_adj_matrix], 
           regex(r"(.+).adjmat.pickle"), 
           [r"\1.%s.region.pickle" % r for r in REGIONS.keys() ])
def carve_out_region(infile, outfiles):

    d = pickle.load(open('data.pickle', 'r'))
    tfdf = d['tfdf']
    wiredf = d['wiredf']
    raw_adj_mat = pickle.load(open(infile, 'r'))['adj_mat']

    for region_i, (region_name, r) in enumerate(REGIONS.iteritems()):
        # select the transitors in the region
        x_mask = (tfdf['x'] >= r['x'][0] ) & (tfdf['x'] <= r['x'][1])
        y_mask = (tfdf['y'] >= r['y'][0] ) & (tfdf['y'] <= r['y'][1])
        idx = x_mask & y_mask
        sub_df = tfdf[idx]
        indices = np.argwhere(idx).flatten()

        # get the resulting indices in the matrix
        sub_mat = raw_adj_mat[indices, :]
        sub_mat = sub_mat[:, indices]
        adj_mat = sub_mat
        # save the positions

        pickle.dump({'infile' : infile, 
                     'region_name' : region_name, 
                     'region' : r, 
                     'subdf' : sub_df, 
                     'indices' : indices, 
                     'adj_mat' : adj_mat}, 
                    open(outfiles[region_i], 'w'))



pipeline_run([load_data, plot_transistors, plot_transistors_regions, 
              create_raw_graph, 
              plot_raw_graph, #analysis, 
              plot_merged_graph, 
              #create_all_adj_matrix, create_dir_adj_matrix, 
              #create_typed_adj_matrix, 
              #create_count_adj_matrix, 
              #plot_adj_matrix, 
              #carve_out_region
          ])

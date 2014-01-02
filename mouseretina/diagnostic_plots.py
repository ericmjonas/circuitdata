import numpy as np
import cPickle as pickle
from matplotlib import pylab
from ruffus import * 
from preprocess import * 
import dbmodel
import pandas
import sqlite3

# plot spatial distribution of each cell type
# plot area vs cell body distance


def create_adj_mat(con, area_thold, cell_data):
    """
    returns (upper-triangular) contact area matrix, cell_ids in order

    """
    
    df = pandas.io.sql.read_frame("select from_id, to_id, area, sum(area) as contact_area, count(area) as contact_count from contacts  where area < %f group by from_id, to_id" % area_thold, 
                                  con)
    
    CELL_N = len(cell_data)
    id_to_pos = {id: pos for pos, id in enumerate(cell_data.index.values)}

    area_mat = np.zeros((CELL_N, CELL_N), dtype=np.float32)

    # for cell_id_1 in cell_data.index.values:
    #     print cell_id_1
    #     for cell_id_2 in cell_data.index.values:
    #         row = df[(df['from_id'] == cell_id_1) & (df['to_id'] == cell_id_2)]
    #         if len(row) > 0:
    #             area_mat[id_to_pos[cell_id_1], 
    #                      id_to_pos[cell_id_2]] = row.iloc[0]['contact_area']
    for c_i, c_row in df.iterrows():
        i1 = id_to_pos.get(c_row['from_id'], -1)
        i2 = id_to_pos.get(c_row['to_id'], -1)
        if i1 >= 0 and i2 >= 0:
            area_mat[i1, i2] = c_row['area']

    lower_triangular_idx = np.tril_indices(CELL_N)

    assert area_mat[lower_triangular_idx].sum() == 0
    return area_mat, cell_data.index.values
            

    
@files("mouseretina.db", "contact_pos.png")
def plot_contacts(infile, outfile):
    """
    scatter plot of all the contact points
    """
    con = sqlite3.connect(infile)
    df = pandas.io.sql.read_frame("select * from contacts where area < %f" % PAPER_MAX_CONTACT_AREA, 
                                  con, index_col='id')

    f = pylab.figure(figsize=(16, 8))

    alpha = 0.05
    s = 1.0
    ax_xz = f.add_subplot(2, 1, 1)
    ax_xz.scatter(df['z'], df['x'], 
                  edgecolor='none', s=df['area']*4, alpha=alpha, c='k')
    ax_xz.set_xlim(0, MAX_DIM[2])
    ax_xz.set_xlabel("z (um)")
    ax_xz.set_ylim(0, MAX_DIM[0])
    ax_xz.set_ylabel("x (um)")

    ax_yz = f.add_subplot(2, 1, 2)
    ax_yz.scatter(df['y'], df['x'], 
                  edgecolor='none', s=df['area']*4, alpha=alpha, c='k')
    ax_yz.set_xlim(0, MAX_DIM[1])
    ax_yz.set_xlabel("y (um)")
    ax_yz.set_ylim(0, MAX_DIM[0])
    ax_yz.set_ylabel("x (um)")

    f.savefig(outfile, dpi=600)

@files("mouseretina.db", "cell_adj.png")
def plot_adj(infile, outfile):
    """
    
    """

    con = sqlite3.connect(infile)
    cell_data = pandas.io.sql.read_frame("select * from cells order by cell_id", 
                                         con, index_col="cell_id")


    area_mat, cell_ids = create_adj_mat(con, PAPER_MAX_CONTACT_AREA)
    area_mat += area_mat.T

    CELL_N = len(cell_ids)
    p = np.random.permutation(CELL_N)
    area_mat_p = area_mat[p, :]
    area_mat_p = area_mat_p[:, p]

    f = pylab.figure(figsize=(8, 8))
    ax = f.add_subplot(1, 1, 1)

    ax.imshow(area_mat_p  > 0.2, interpolation='nearest', 
              cmap=pylab.cm.Greys)
    
    f.savefig(outfile, dpi=600)

@files("mouseretina.db", 
       ['somapos.pdf'])
def plot_somapos(infile, (pos_outfile,)):
    """
    Plot the physical positions of the cells, 
    colored by "coarse type"
    """

    con = sqlite3.connect(infile)
    cells = pandas.io.sql.read_frame("select c.cell_id, s.x, s.z, s.y, t.coarse from cells as c join somapositions as s on c.cell_id = s.cell_id join types as t on c.type_id = t.type_id", 
                                     con, index_col='cell_id')
    
    color_map = {'gc' : 'r', 
                 'nac' : 'b', 
                 'mwac' : 'y', 
                 'bc' : 'g', 
                 'other' : 'k', 
                 None : 'k'}

    colors = [color_map[c] for c in cells['coarse']]
        
    f = pylab.figure(figsize=(16, 8))
    ax = f.add_subplot(2, 1, 1)
    S = 20
    alpha = 0.7
    ax.scatter(cells['y'], cells['x'], c=colors, 
               edgecolor='none', s=S, alpha=alpha)
    ax.set_xlim(0, 120)
    ax.set_xlabel('y (um)')
    ax.set_ylabel('x (um)')

    ax = f.add_subplot(2, 1, 2)
    ax.scatter(cells['z'], cells['x'], c = colors, 
               edgecolor='none', s=S, alpha=alpha)
    ax.set_xlim(0, 120)
    
    ax.set_xlabel('z (um)')
    ax.set_ylabel('x (um)')

    f.savefig(pos_outfile)

EXAMPLES = [10, 50, 100, 150, 200, 250, 300, 350, 400]
@files(['soma.positions.pickle', 'synapses.pickle'], 
       ['example.%d.pdf' % e for e in EXAMPLES])
def plot_example_cells((pos_file, synapse_file), output_files):
    soma_pos = pickle.load(open(pos_file, 'r'))
    synapses = pickle.load(open(synapse_file, 'r'))['synapsedf']

    for CELL_ID, output_file in zip(EXAMPLES,  output_files):
        soma_pos_vec = soma_pos['pos_vec']


        f = pylab.figure(figsize=(16, 8))
        ax_yx = f.add_subplot(2, 1, 1)
        ax_yx.scatter(soma_pos_vec[:, 1], soma_pos_vec[:, 0], 
                      c='k', edgecolor='none', s=3)
        ax_yx.set_xlim(120, 0)
        ax_yx.set_ylim(160, 0)
        ax_yx.set_xlabel('y (um)')
        ax_yx.set_ylabel('x (um)')

        ax_zx = f.add_subplot(2, 1, 2)
        ax_zx.scatter(soma_pos_vec[:, 2], soma_pos_vec[:, 0], 
                      c='k', edgecolor='none', s=3)
        ax_zx.set_xlim(80, 0)
        ax_zx.set_ylim(160, 0)
        ax_zx.set_xlabel('z (um)')
        ax_zx.set_ylabel('x (um)')

        # now plot the target cell
        tgt = soma_pos_vec[CELL_ID]
        ax_yx.scatter(tgt[1], tgt[0], c='r', s=20)
        ax_zx.scatter(tgt[2], tgt[0], c='r', s=20)

        tgt_df = synapses[(synapses['from_id'] == CELL_ID) | (synapses['to_id'] == CELL_ID)]
        for area_range, size, color in [((0.0, 0.1), 1, 'b'), 
                                        ((0.1, 1.0), 5, 'g'), 
                                        ((1.0, 5.0), 15, 'r')]:
            plot_df = tgt_df[(tgt_df['area'] > area_range[0]) & (tgt_df['area'] < area_range[1])]
            ax_yx.scatter(plot_df['y'], plot_df['x'], s=size, c=color, 
                          edgecolor='none', alpha=0.5)
            ax_zx.scatter(plot_df['z'], plot_df['x'], s=size, c=color, 
                          edgecolor='none', alpha=0.5)

        f.savefig(output_file)

@files(["type_metadata.pickle", "soma.positions.pickle", "xlsxdata.pickle"], 
       ['adjmat.byclass.png'])
def plot_adjmat_byclass((type_file, pos_file, xlsxdata_file), (adj_mat_plot,)):
    soma_pos = pickle.load(open(pos_file, 'r'))
    type_metadata = pickle.load(open(type_file, 'r'))['type_metadata']
    xlsdata = pickle.load(open(xlsxdata_file, 'r'))
    types = xlsdata['types']
    area_mat = xlsdata['area_mat']
    pos_vec = soma_pos['pos_vec']
    CELL_N = len(pos_vec)
    print "CELL_N=", CELL_N


    f = pylab.figure(figsize=(12, 12))
    ax = f.add_subplot(1, 1, 1)
    
    x_pos = []
    y_pos = []
    s = []
    for i in range(CELL_N):
        for j in range(CELL_N):
            a = area_mat[i, j]
            if a > 1e-6:
                x_pos.append(i)
                y_pos.append(j)
                s.append(a)
    ax.scatter(x_pos, y_pos, s=s, edgecolor='none', alpha=0.5, c='k')
    ax.set_xlim(0, CELL_N)
    ax.set_ylim(CELL_N, 0)
    ax.set_title("mouse retina connectivity matrix")
    f.tight_layout()
    f.savefig(adj_mat_plot, dpi=200)


@files(["type_metadata.pickle", "soma.positions.pickle", "xlsxdata.pickle"], 
       ['adjmat.byz.png'])
def plot_adjmat_byz((type_file, pos_file, xlsxdata_file), (adj_mat_plot,)):
    soma_pos = pickle.load(open(pos_file, 'r'))
    type_metadata = pickle.load(open(type_file, 'r'))['type_metadata']
    xlsdata = pickle.load(open(xlsxdata_file, 'r'))
    types = xlsdata['types']
    area_mat = xlsdata['area_mat']
    pos_vec = soma_pos['pos_vec']
    CELL_N = len(pos_vec)
    print "CELL_N=", CELL_N
    ai =  np.argsort(pos_vec[:, 2]).flatten()
    area_mat = area_mat[ai]
    area_mat = area_mat[:, ai]

    
    f = pylab.figure(figsize=(12, 12))
    ax = f.add_subplot(1, 1, 1)
    
    x_pos = []
    y_pos = []
    s = []
    for i in range(CELL_N):
        for j in range(CELL_N):
            a = area_mat[i, j]
            if a > 1e-6:
                x_pos.append(i)
                y_pos.append(j)
                s.append(a)
    ax.scatter(x_pos, y_pos, s=s*2, edgecolor='none', alpha=0.5, c='k')
    ax.set_xlim(0, CELL_N)
    ax.set_ylim(CELL_N, 0)
    ax.set_title("mouse retina connectivity matrix")
    f.tight_layout()
    f.savefig(adj_mat_plot, dpi=200)



pipeline_run([
              plot_contacts, 
    plot_somapos, 
    plot_adj, 
    #plot_example_cells, plot_adjmat_byclass, 
    #plot_adjmat_byz
])

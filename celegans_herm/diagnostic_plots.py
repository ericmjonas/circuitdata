from ruffus import *
import cPickle as pickle
import numpy as np
import pandas
import copy
import os, glob
import time
from matplotlib import pylab
from jinja2 import Template
import irm
import irm.data
import matplotlib.gridspec as gridspec
from matplotlib import colors
import copy
import sqlite3



@files('celegans_herm.db', 'class.positions.pdf')
def plot_positions(infile, outfile):
    conn = sqlite3.connect(infile)
    
    n = pandas.io.sql.read_frame("select * from Cells", 
                                 conn, index_col="cell_name")
    
    f = pylab.figure()
    ax = f.add_subplot(1, 1, 1)
    plotted = 0
    for ei, (gi, g) in enumerate(n.groupby('cell_class')):
        N = len(g)
        if N >= 4:
            r = g.iloc[0]['role']
            if r == 'M':
                c = 'b'
            elif r == 'S':
                c = 'g'
            else:
                c = "#AAAAAA"
            ax.plot([0, 1], [plotted, plotted], c='k', alpha=0.5, linewidth=1)
            ax.scatter(g['soma_pos'], np.ones(N) * plotted, 
                       c=c, s=30)
            ax.text(1.01, plotted - 0.15, gi, fontsize=10)
            plotted +=1

    ax.set_xlim(-0.05, 1.1)
    ax.set_xlabel("position along anterior-posterior axis")
    ax.set_ylim(-1, plotted)
    ax.set_yticks([])
    ax.set_title('c elegans cell class positions')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')

    f.savefig(outfile)

@files('celegans_herm.db', 'class.adjmat.pdf')
def plot_adjmat(infile, outfile):
    conn = sqlite3.connect(infile)
    
    n = pandas.io.sql.read_frame("select * from Cells order by soma_pos", 
                                 conn, index_col="cell_id" )
    n['index'] = np.arange(len(n), dtype=np.int)
    
    s = pandas.io.sql.read_frame("select * from Synapses", 
                                 conn)
    idx = n['index']
    print idx
    s['from_idx'] = np.array(idx[s['from_id']])
    s['to_idx'] = np.array(idx[s['to_id']])
    
    
    f = pylab.figure(figsize=(6, 6))
    ax = f.add_subplot(1, 1, 1)
    chem = s[s['synapse_type'] == 'C']
    elec = s[s['synapse_type'] == 'E']

    ax.scatter(chem['from_idx'], chem['to_idx'], c='b', 
               s=chem['count']*2, 
               edgecolor='none', alpha=0.5)
    ax.scatter(elec['from_idx'], elec['to_idx'], c='r', 
               s=elec['count']*2, 
               edgecolor='none', alpha=0.5)
    
    ax.set_xlim(0, 279)
    ax.set_ylim(279, 0)


    f.savefig(outfile)


pipeline_run([ plot_positions, 
               plot_adjmat, 
               #plot_adjmat, plot_classmat,
              #plot_conn_dist
           ])

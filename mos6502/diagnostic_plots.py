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


@files('mos6502.db', "transistors.positions.pdf")
def plot_transistors(input_file, output_file):

    conn = sqlite3.connect(input_file)
    
    tfdf = pandas.io.sql.read_frame("select * from Transistors", 
                                 conn, index_col="name")
    
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


if __name__ == "__main__":
    pipeline_run([plot_transistors])

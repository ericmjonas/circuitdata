import numpy as np
import cPickle as pickle
from matplotlib import pylab
from xlrd import open_workbook
import pandas
import os
from ruffus import * 
import dbmodel
import subprocess

DATA_DIR = "../../data/celegans/conn2"

@files([os.path.join(DATA_DIR, x) for x in ["NeuronConnect.xls", "NeuronType.xls", '../manualmetadata.xlsx']], "data.pickle")
def read_data((neuron_connect_file, neuron_type_file, 
               manual_metadata_file), output_file):


    neuron_connect =  open_workbook(neuron_connect_file)
    connections = {}
    for s in neuron_connect.sheets():
        for row_i in range(1, s.nrows):
            neuron_1 = s.cell(row_i, 0).value
            neuron_2 = s.cell(row_i, 1).value
            typ = s.cell(row_i, 2).value
            nbr = int(s.cell(row_i, 3).value)

            c = neuron_1, neuron_2
            if c not in connections:
                connections[c] = []
            connections[c].append((typ, nbr))

    neurons = {}
    neuron_connect =  open_workbook(neuron_type_file)
    for s in neuron_connect.sheets():

        for row_i in range(1, s.nrows):
            neuron = str(s.cell(row_i, 0).value)
            soma_pos = float(s.cell(row_i, 1).value)
            neurons[neuron] = {'soma_pos' : soma_pos}

    # now open the excel workbook of our metadata
    metadata_df = pandas.io.excel.read_excel(manual_metadata_file, 
                                             'properties', 
                                             index_col=0)
    assert len(metadata_df) == len(neurons)
    ndf = pandas.Series([d['soma_pos'] for d in neurons.values()], 
                        neurons.keys())
    metadata_df['soma_pos'] = ndf
    n = metadata_df

    # find classes
    classes = {}
    orig_names = set(n.index)
    orig_names_base = set(orig_names)
    for cur_n in orig_names_base:
        #cur_n = orig_names.pop()
        # does its name end in two digits? That's your class

        if cur_n not in orig_names:
            continue
        if cur_n[-2:].isdigit():
            base = cur_n[:-2] 
            tgts = [s for s in orig_names if s.startswith(base) and s[-2:].isdigit()]
            classes[base] = tgts
            orig_names -= set(tgts)

    orig_names_base = set(orig_names)
    for cur_n in orig_names_base:
        if cur_n not in orig_names:
            continue
        if cur_n[-2:] == 'DL':
            base = cur_n[:-2]
            print "base =", base
            if base+"VL" in orig_names:
                # is this six way? 
                tgts = [base + sub for sub in ['DL', 'DR', 'VL', 'VR', 'L', 'R']]
                if set(tgts).issubset(orig_names):
                    # we are good to go
                    classes[base] = tgts
                    orig_names -= set(tgts)
                # four way
                tgts = [base + sub for sub in ['DL', 'DR', 'VL', 'VR']]
                if set(tgts).issubset(orig_names):
                    # we are good to go
                    classes[base] = tgts
                    orig_names -= set(tgts)
                

    orig_names_base = set(orig_names)
    for cur_n in orig_names_base:
        if cur_n not in orig_names:
            continue
        if cur_n[-1] == 'R' and cur_n[-2] != 'V':
            base = cur_n[:-1]
            if base+"L" in orig_names:
                tgts = [base + 'R', base+'L']
                classes[base] = tgts
                orig_names -= set(tgts)
    # the singletons
    for o in orig_names:
        classes[o] = [o]
    print 'classes', classes['IL1']

    n_s = []
    c_s = []
    for c, ns in classes.iteritems():
        for neuron in ns:
            n_s.append(neuron)
            c_s.append(c)
    s= pandas.Series(c_s, index=n_s)
    n['class']=s        

    pickle.dump({'connections' : connections, 
                 'neurons' : n}, 
                open(output_file, 'w'))

@files(read_data, "report.txt")
def sanity_check(infile, outfile):
    """ Sanity check to make sure we understand
    the semantic meaning of the type codes
    """

    indata = pickle.load(open(infile, 'r'))
    neurons = indata['neurons']
    connections = indata['connections']



    # electrical synapses are encoded symmetrically
    elec_conn = set()
    for (n1, n2), conns in connections.iteritems():
        for c_t, v in conns:
            if c_t == 'EJ':
                elec_conn.add((n1, n2))
    for n1, n2 in elec_conn:
        assert (n2, n1) in elec_conn

    # # Are the "poly" checks consistent? 
    # # if neuron A is Rp, does it really receive inputs from more than one
    # # neuron? 

    # # if neuron A is just R (no p), does it only receive input from a 
    # # single neuron

    # # if neuron A is Sp...

    # # if neuron A is just S

    # # does every S match with a R ? 
    
    s_conn = {}
    r_conn = {}
    sp_conn = {}
    rp_conn = {}
    for (n1, n2), conns in connections.iteritems():
        for c_t, v in conns:
            if c_t == 'S':
                s_conn[(n1, n2)] = v
            elif c_t == 'R':
                r_conn[(n1, n2)] = v
            elif c_t == 'Sp':
                sp_conn[(n1, n2)] = v
            elif c_t == 'Rp':
                rp_conn[(n1, n2)] = v
    
                



    
@files(read_data, dbmodel.DB_NAME)
def populate(infile, outfile):
    dbmodel.db.connect()
    dbmodel.create_db()
    from dbmodel import Cells, Synapses
    
    # load the individual neurons
    indata = pickle.load(open(infile, 'r'))
    neurons = indata['neurons']

    for neuron_name, nd in neurons.iterrows():
        nt = nd['neurotransmitters']
        if type(nt) == float:
            nt = None

        role = nd['role']
        if type(role)==float:
            role = None
        Cells.create(cell_name =neuron_name, 
                     cell_class= nd['class'], 
                     soma_pos = nd['soma_pos'], 
                     neurotransmitters=nt, 
                     role = role)
        
    connections = indata['connections']
    
    for n1 in neurons.index.values:
        for n2 in neurons.index.values:
            c1 = Cells.get(Cells.cell_name == n1)
            c2 = Cells.get(Cells.cell_name == n2)

            c = (n1, n2)
            if c in connections:
                for synapse_type, nbr in connections[c]:
                    code = synapse_type[0]
                    st_short = None
                    # note this dataset includes entries for both A sends to B (type S) and B receives from A (type R)
                    if code == "Sp" or code == 'S':
                        st_short = 'C'
                    elif code == 'E':
                        st_short = 'E'
                    elif code == 'R' or code == 'Rp':
                        pass
                    else:
                        raise Exception("code %s not understood" % code)
                    if st_short:
                        s = Synapses.select().where(Synapses.from_id == c1, 
                                                    Synapses.to_id == c2, 
                                                    Synapses.synapse_type == st_short).count()
                        if s == 0:

                            Synapses.create(from_id = c1,
                                            to_id = c2, 
                                            synapse_type = st_short,
                                            count = nbr)
                        else:
                            Synapses.update(count = Synapses.count + nbr).where(Synapses.from_id == c1, 
                                                                                Synapses.to_id == c2, 
                                                                                synapse_type == st_short)
    dbmodel.db.close()

pipeline_run([read_data, sanity_check, populate, 
              ])

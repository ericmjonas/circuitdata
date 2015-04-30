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
import dbmodel 

DATA_DIR = "../../data/netlists"
@files(os.path.join(DATA_DIR, "data.json"), 'data2.pickle')
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

    pickle.dump({'tfdf' : tfdf, 'wiredf'  : df}, 
                open(output_file, 'w'))

@files(load_data, dbmodel.DB_NAME)
def populate(infile, outfile):
    dbmodel.db.connect()
    dbmodel.create_db()
    from dbmodel import Transistors, Wires

    indata = pickle.load(open(infile, 'r'))

    for wire_id, wd in indata['wiredf'].iterrows():
        Wires.create(id=wire_id, 
                     c1c2s = wd['c1c2s'], 
                     gates = wd['gates'], 
                     pullup = wd['pullup'], 
                     name = wd['name'])


    for transistor_name, td in indata['tfdf'].iterrows():
        Transistors.create(name=transistor_name, on = td['on'], 
                           gate = td['gate'], 
                           c1 = td['c1'], c2 = td['c2'], 
                           bb_x1 = td['bb'][0], 
                           bb_x2 = td['bb'][1], 
                           bb_y1 = td['bb'][2], 
                           bb_y2 = td['bb'][3], 
                           x = td['x'], 
                           y = td['y'])



    dbmodel.db.close()


if __name__ == "__main__":
    pipeline_run([load_data, populate])


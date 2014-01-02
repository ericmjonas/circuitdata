from ruffus import *
from matplotlib import pylab
from xlrd import open_workbook
import glob
import os
import pandas


import cPickle as pickle
import numpy as np
import scipy.io
import skimage 
import skimage.draw
import skimage.feature
import skimage.morphology
import skimage.measure
import skimage.io
import dbmodel 

LIGHT_AXIS = [0.9916,0.0572, 0.1164]
MAX_DIM = [132.0, 114.0, 80.0]
BASEDIR = "../../data/mouseretina" 
PAPER_MAX_CONTACT_AREA = 5.0 # microns, to eliminate touching somata



@files("%s/Helmstaedter_et_al_SUPPLinformation5.mat" % BASEDIR, 
       "synapses.pickle")
def load_synapse_data(mat_file, output_file):
    """
    raw synapse data from the matlab file. 
    zero-center
    """
    
    d = scipy.io.loadmat(mat_file)
    
    data = d['kn_allContactData_Interfaces_duplCorr_output_IDconv']

    raw = []
    for from_id, to_id, area, x, y, z in data:
        # ASSUMPTION : from < to, upper right-hand of matrix
        if from_id > to_id:
            from_id, to_id = to_id, from_id

        raw.append((int(from_id), int(to_id), x/1000., y/1000., z/1000., area))
    df = pandas.DataFrame.from_records(raw, columns=['from_id', 'to_id', 'x', 'y', 'z', 'area'])
    pickle.dump({'synapsedf' : df},  
                 open(output_file, 'w'))


@files("%s/Helmstaedter_et_al_SUPPLinformation4.xlsx" % BASEDIR, 
       "xlsxdata.pickle")
def load_xlsx_data(xlsx_file, output_file):
    neuron_connect =  open_workbook(xlsx_file)
    connections = {}
    s =  neuron_connect.sheets()[0]
    N = 1123
    data = np.zeros((N, N), dtype=np.float32)
    for cell_i in range(N):
        for cell_j in range(N):
            syn_area = s.cell(cell_i + 1, cell_j+1).value
            data[cell_i, cell_j] = syn_area

    s = neuron_connect.sheets()[2]
    types = {}
    for cell in range(N):
        cell_id = s.cell(cell+1, 0).value
        type_id = s.cell(cell +1, 3).value
        types[cell+1] = type_id
    
    pickle.dump({'area_mat' : data, 
                 'types' : types}, 
                open(output_file, 'w'))

@files([load_synapse_data, load_xlsx_data], 
       ["conn.areacount.pickle"])
def transform_data((synapse_file, xlsx_data_file),  (areacount_file,)):
    """
    use the xlsdata file to copy over the type information because
    we want it all in this master "input" file
    """
    synapses = pickle.load(open(synapse_file, 'r'))['synapsedf']
    HIGHEST_CELL_ID = synapses['to_id'].max()
    CELL_N = HIGHEST_CELL_ID + 1


    area_mat = np.zeros((CELL_N, CELL_N), dtype=[('area', np.float32), 
                                                 ('count', np.uint32)])

    for (from_id, to_id), cell_synapses in synapses.groupby(['from_id', 'to_id']):
        area_mat[from_id - 1, to_id - 1]['count'] = len(cell_synapses)
        area = cell_synapses['area']

        area_mat[from_id - 1, to_id - 1]['area'] = area[area < PAPER_MAX_CONTACT_AREA].sum()

    # now make symmetric
    lower_idx = np.tril_indices(CELL_N)
    # now this should be zeros
    assert area_mat[lower_idx]['count'].sum() == 0
    area_mat['count'] += area_mat.T['count']
    area_mat['area'] += area_mat.T['area']

    xlsx_data = pickle.load(open(xlsx_data_file, 'r'))
    
    pickle.dump({'area_mat': area_mat, 
                 'types' : xlsx_data['types']}, 
                open(areacount_file, 'w'))

def cell_image_files():

    for directory in [os.path.join(BASEDIR, "nature12346-s6/Supp_Info6a/*.png"), 
                      os.path.join(BASEDIR, "nature12346-s6/Supp_Info6b/*.png"), 
                      os.path.join(BASEDIR, "nature12346-s7/*.png"), 
                      os.path.join(BASEDIR, "nature12346-s8/*.png"), 
                      os.path.join(BASEDIR, "nature12346-s9/Supp_Info6e/*.png"), 
                      os.path.join(BASEDIR, "nature12346-s9/Supp_Info6f/*.png")]:
        files = glob.glob(directory)
        for filename in files:
            f = os.path.basename(filename)
            cell_id = int(f[5:9])
            out_basename = "imgproc/%04d" % cell_id
            yield filename, [out_basename + '.png', out_basename + ".pickle"]

@follows(mkdir("imgproc"))
@files(cell_image_files)
def process_image_pos(filename, (output_png, output_pickle)):

    W = 200
    H = 200
    R = 50
    template = np.zeros((H, W))
    a = skimage.draw.circle(H/2, W/2, R, (H, W))
    template[a] = 1
    
    REGION_X = 0, 1900
    REGION_Y = 0, 3800

    results = {}
    

    x = skimage.io.imread(filename)
    x_sub = x[REGION_Y[0]:REGION_Y[1], REGION_X[0]:REGION_X[1]]

    tgt = (x_sub[:, :, 0] < 200) & (x_sub[:, :, 1] < 200) & (x_sub[:, :, 2] < 240) & (x_sub[:, :, 2] >100)

    m = skimage.feature.match_template(tgt, template, pad_input=True)       

    tholded = m > 0.22
    label_image = skimage.morphology.label(tholded)

    coords_x_y = []
    for region in skimage.measure.regionprops(label_image, ['Area', 'Centroid']):

        # skip small images

        if region['Area'] < 1000:
            continue
        c = region['Centroid']

        coords_x_y.append((c[1], c[0]))

    if len(coords_x_y) != 4:
        print "DANGER ERROR", coords_x_y
        pylab.subplot(1, 2, 1)
        pylab.imshow(tholded)
        pylab.subplot(1, 2, 2)
        pylab.imshow(x_sub)
        pylab.show()
        raise RuntimeError("found %d coords in file %s" % (len(coords_x_y), filename))

    results = {'filename' : filename, 
               'coords' : coords_x_y}
    pickle.dump(results, open(output_pickle, 'w'))
    
    f = pylab.figure()
    ax  = f.add_subplot(1, 1, 1)
    ax.imshow(x_sub)
    for c in coords_x_y:
        ax.scatter(c[0], c[1], c='r')
    f.savefig(output_png)
    f.clf()
    del f

@merge(process_image_pos, "soma.positions.pickle")
def merge_positions(inputfiles, outputfile):
    """
    Remember we have these as pixes, so we need to subtract and organize. 

    """

    PIX_PER_UM = 7.2
    # use the top two plots
    out_pos = {}
    out_coords = {}
    N = len(inputfiles)
    coords = np.zeros((N, 3), dtype=np.float32)
    LEFT_1 = 100
    LEFT_2 = 1090
    ZERO_OFFSET = 70
    for f in inputfiles:
        cell_id = int(os.path.basename(f[1])[:4])
        d = pickle.load(open(f[1]))

        # NOTE THESE ARE IN BS UNITS
        x = 0
        y = 0
        z = 0
        for c in d['coords']:
            print "The coords are", c
            if c[0] < LEFT_2 and c[1] < 900:
                # upper left plot in image, meaning u-z
                x = (c[1] + ZERO_OFFSET) / PIX_PER_UM
                y = MAX_DIM[1] - ((c[0] - LEFT_1) / PIX_PER_UM)
            elif c[0] > LEFT_2 and c[1] < 900:
                x = (c[1] + ZERO_OFFSET) / PIX_PER_UM
                z = MAX_DIM[2] - (c[0] - LEFT_2) / PIX_PER_UM
        
        if x == 0 or y == 0 or z == 0:
            raise Exception("did not find one!")
        out_coords[cell_id] = d['coords']
        out_pos[cell_id] = (x, y, z)

    pickle.dump({'pos' : out_pos, 
                 'coords_px' : out_coords,
             }, open(outputfile, 'w'))
    
@files("%s/types.xlsx" % BASEDIR,
       "type_metadata.pickle")
def type_metadata(xlsx_file, output_file):
    neuron_connect =  open_workbook(xlsx_file)

    s =  neuron_connect.sheets()[0]
    N = 71
    res = []
    
    for type_i in range(N):
        type_id = int(s.cell(type_i + 1, 0).value)
        desig = s.cell(type_i + 1, 1).value
        volgyi = s.cell(type_i + 1,2).value
        macneil = s.cell(type_i + 1, 3).value
        certainty = s.cell(type_i + 1, 4).value
        if volgyi != "":
            other = volgyi
        else:
            other=macneil
        res.append({'id' : type_id, 
                    'desig' : desig, 
                    'other' : other, 
                    'macneil' : macneil, 
                    'volgyi' : volgyi, 
                    'certainty' : certainty})
    df = pandas.DataFrame(res)
    df = df.set_index(df['id'])

    # the following preprocessing information was acquired from
    # page 1 of the s1
    df['coarse'] = df['desig'].apply(lambda x: "other")
    df['coarse'][(df['id'] <=12)] = "gc"
    df['coarse'][(df['id'] > 12) & (df['id'] <= 24)] = "nac"
    df['coarse'][(df['id'] > 24) & (df['id'] <= 57)] = "mwac"
    df['coarse'][(df['id'] > 58) & (df['id'] <= 71)] = "bc"
    
    del df['id']
                    
    pickle.dump({'type_metadata' : df}, 
                open(output_file, 'w'))
@files(None, dbmodel.DB_NAME)
def create_db(_, outfile):

    dbmodel.db.connect()

    dbmodel.create_db()

@follows(load_xlsx_data)
@follows(type_metadata)
@follows(merge_positions)
@follows(create_db)
def populate_db():
    dbmodel.db.connect()
    
    from dbmodel import Types, Cells, SomaPositions, Contacts

    with dbmodel.db.transaction():
        types_df = pickle.load(open('type_metadata.pickle', 'r'))['type_metadata']
        for r_id, r in types_df.iterrows():
            print "rid=", r_id
            t = Types.create(type_id = int(r_id), 
                             designation = r['desig'], 
                             volgyi_type = r['volgyi'], 
                             macneil_type = r['macneil'], 
                             certainty = r['certainty'], 
                             coarse = r['coarse'])
            t.save()
        # create the additional types, because... this data is noisy
        for i in range(72, 79):
            t = Types.create(type_id = i)

        cell_types = pickle.load(open('xlsxdata.pickle', 'r'))['types']

        for c_i, c_t in cell_types.iteritems():
            c = Cells.create(cell_id = c_i, 
                             type_id = Types.get(Types.type_id == c_t))
            c.save()
    
                          
        pos = pickle.load(open("soma.positions.pickle", 'r'))
        for k, v in pos['pos'].iteritems():
            p = SomaPositions.create(cell_id = Cells.get(Cells.cell_id ==k), 
                                     x = v[0], y =v[1], z=v[2])
            p.save()
        
        s = pickle.load(open("synapses.pickle", 'r'))
        for r_id, r in s['synapsedf'].iterrows():
            for from_id in Cells.select().where(Cells.cell_id == r['from_id']):
                for to_id in Cells.select().where(Cells.cell_id == r['to_id']):

                    # WE ARE INTENTIONALLY THROWING OUT ALL THE CONTACTS
                    # that have cell_ids that don't map to known cells. 
                    Contacts.create(from_id = from_id, to_id = to_id, 
                                    x = r['x'], y=r['y'], z=r['z'], 
                                    area = r['area']).save()
            
@follows(transform_data)                                                      
@files(["conn.areacount.pickle", "xlsxdata.pickle"], 
       "adj_comp.pdf")
def mat_xls_consistency((conn_areacount, xlsdata), adj_plots):
    """
    To what degree does the excel data and  the matlab data 
    agree? 

    """
    xls_d = pickle.load(open(xlsdata, 'r'))
    mat_d = pickle.load(open(conn_areacount, 'r'))

    print "CELLS IN XLS:", len(xls_d['area_mat'])
    print "CELLS IN MATLAB:", len(mat_d['area_mat'])
    CELL_N = 950
    xls_m = xls_d['area_mat'][:CELL_N, :CELL_N]
    mat_m = mat_d['area_mat'][:CELL_N, :CELL_N]['area']

    print xls_m[:3, :3]
    print mat_m[:3, :3]

    f = pylab.figure()
    ax_delta = f.add_subplot(1, 1, 1)
    delta = np.abs(xls_m - mat_m)
    delta_signed = xls_m - mat_m
    # print "There are", np.sum(delta > 0.001), "disagreeing cells"
    # for delta_pos in np.argwhere(delta > 0.001):
    #     p0 = delta_pos[0]
    #     p1 = delta_pos[1]
    #     #print "at ", delta_pos, "the error is", delta_signed[p0, p1]
    #     sc = mat_d['area_mat'][p0, p1]['count']
    #     #print "synapse count =", sc
            
    ax_delta.hist(delta_signed, bins=20)
    ax_delta.set_xlabel("spreadsheet - mat")
    ax_delta.set_ylabel("count")
    ax_delta.set_title("Difference between xls and mat")

    f.savefig(adj_plots)
    
    

        
if __name__ == "__main__":
    
    pipeline_run([load_synapse_data, load_xlsx_data, transform_data, 
                  #sanity_check, plot_synapses, 
                  mat_xls_consistency, 
                  process_image_pos, merge_positions, type_metadata, 
                  create_db, 
                  populate_db
              ])


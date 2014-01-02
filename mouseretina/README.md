Mouse Retina Connectome
============================

To generate this file, we process the original provided .mat file, 
the supplemental information spreadsheet, we extract soma
positions from the provided collection of PNGs, and we 
get cell type metadata from the provided PDF of cell types
(That we manually turned into a spreadsheet)

Note that we derive soma positions by reverse-engineering the PNGs
provided and thus 1. only have soma positions for a subset of the
cells and 2. are making some assumptions about the correct spatial scale
(partly by visual inspection to see if the locations of the synapses
/dendritic arbors in the synapses table "match up" with the soma pos --
we plot and compare to the provided PNGs)

Schema
=======

CELLS : table of cell ID and type
cell_id: globally-unique cell ID, from original paper ID
type_id : type_id, references TYPES table

SomaPositions : The locations of the soma 
cell_id
x
y
z

Contacts: The locations of the cell-cell contact points
 (this is validated between xls and the matlab file)
from_id: 
to_id:
x
y
z
area

TYPES: Cell Type metadata, reverse-engineered from their plot; only covers
types 1-71 (why are there more types? what are they?)

type_id : type ID
designation
Volgyi_volgi : type from CITE
MacNeil: Type from Macneil
certainty : The author's arbitrary uncertainty metric of L, M, H
coarse : the coarse type 

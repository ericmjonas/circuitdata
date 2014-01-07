Mouse Retina Connectome
============================

The database is ```mouseretina.db.gz```

This is data from [Connectomic reconstruction of the inner plexiform
layer in the mouse
retina](http://www.nature.com/nature/journal/v500/n7461/full/nature12346.html)
by Moritz Helmstaedter, Kevin L. Briggman,,Srinivas C. Turaga, Viren
Jain, H. Sebastian Seung and Winfried Denk (08 August 2013)

 
To generate this database, we process the original provided .mat file, 
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

### CELLS

Table of cell ID and type

* ```cell_id```: globally-unique cell ID, from original paper ID
* ```type_id```: type_id, references Types table

### SomaPositions

The locations of the soma of the cells, from original plots, 
via image reconstruction. Only available for 950 cells. 

* ```cell_id```: relates to cells 
* ```x``` : position (um)
* ```y``` : position (um)
* ```z``` : position (um )

###Contacts

The locations of the cell-cell contact points (this is validated between xls and the matlab file). Remember the edges are undirected, so from/to are arbitrary

* ```from_id```: cell id of first cell
* ```to_id```: cell id of second cell
* ```x``` : position of contact (um)
* ```y``` : position of contact (um)
* ```z``` : position of contact (um)
* ```area``` : area of contact (um^2)


### Types

 Cell Type metadata, reverse-engineered from their plot; only covers
types 1-71 (why are there more types? what are they?)

* ```type_id``` : type ID
* ```designation```: the authors' original string designating the cells
* ```Volgyi``` : type from CITE
* ```MacNeil```: Type from Macneil
* ```certainty``` : The author's arbitrary uncertainty metric of L, M, H when relating the cell to one of the known types
* ```coarse``` : the coarse type, determined from supplemental text

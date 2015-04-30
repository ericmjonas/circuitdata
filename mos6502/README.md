MOS 6502 Circuit
============================

The database is ```mos6502.db``` and all of the data was obtained
via [visual6502](http://visual6502.org/ ) We downloaded the underlying
json and parsed it.

There are actually two preprocessing scripts here,
``preprocess.py``
and
``preprocess_old.py``

The latter was actually used in the paper and simply saves the raw files
out as pickled pandas dataframes after performing various bits of graph
munging to merge nodes and wires. 

Schema
=======

### WIRES

Wires connect transistors, count the number of each edge type, and potentially
have a human-interpretable name. 

* ```id```: globally-unique wire ID, from original dataset
* ```c1c2s```: number of c1/c2 terminals connected to this wire
* ```gates```: number of gates connected to this wire
* ```pullup```: is this wire pulled up ? 
* ```name```: human-useful name of wire


### TRANSISTORS
These are the transistors

* ```name```: globally-unique transistor name, of the form tnnnn
* ```on```: is the transistor on? 
* ```gate```: wire connected to gate (refers to WIRES.id)
* ```c1```: wire connected to c1 (refers to WIRES.id)
* ```c2```: wire connected to c2 (refers to WIRES.id)
* ```bb_x1```: Bounding box x1
* ```bb_x2```: Bounding box x2
* ```bb_y1```: Bounding box y1
* ```bb_y2```: Bounding box y2
* ```x``` : center of bounding box, x direction -- used as transistor "position"
* ```y``` : center of bounding box, y direction -- used as transistor "position"






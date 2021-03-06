Anterior Herm. C. Elegans Connectome
====================================

The database is ```celegans_herm.db.gz```

The connectome of the anterior section of the hermaphrodite c. elegans. The source
of this data is from the [second listed dataset](http://www.wormatlas.org/neuronalwiring.html#NeuronalconnectivityII) at wormatlas. Quoting from them:

> This data was first discussed by Chen, Hall, and Chklovskii, in "Wiring 
optimization can relate neuronal structrure and function", PNAS, March 21, 2006 103: 4723-4728 [doi:10.1073/pnas.0506806103](http://www.pnas.org/content/103/12/4723.abstract). More recently, full analysis of the 
data can be found by Varshney, Chen, Paniaqua, Hall and Chklovskii in "Structural 
properties of the C. elegans neuronal network" PLoS Comput. Biol. Feb 3, 
2011 3:7:e1001066 [doi:10.1371/journal.pcbi.1001067](http://www.ploscompbiol.org/article/info%3Adoi%2F10.1371%2Fjournal.pcbi.1001066). Any publication made utilizing the following data should make a reference to this paper.

I am not incredibly thrilled with the quality of the role and neurotransmitter
metadata for this model. With as much research as has been put into this
system over the years, it's frustrating that bits of the worm atlas aren't even
consistent. Nonetheless, this is a reasonable first attempt, especilaly for 
multi-cell classes. 

### Cells

* cell_id : auto-inc cell ID
* cell_name : official cell name, from original brenner paper
* cell_class : an attempt at determining the cell class
* soma_pos : position along the body axis, range : [0, 1]
* role: text string of Motor, sensory, interneuron 
* neurotransmitter: text string of type of neurotransmitter

### Synapses

Synapses; note that the electrical synapses are undirected, the chemical synapses directed. 

* from_id : from cell id
* to_id : to cell id
* synapse_type: 'E' for electrical, 'C' for chemical 
* count : # of this kind of synapse


TODO
------
- [ ] Add "gross types"/ role from metadata
- [ ] more Diagnostic plots


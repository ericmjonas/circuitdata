Connectomics Data Sets
========================

The purpose of this archive is to provide canonical, cleaned-up,
properly normalized versions of various "computing systems"
datasets. All files are provided in SQLite format. This arose due to
the difficulty of getting both clean data and metadata when working
with these data sets -- dear scientists, please stop releasing zip
files of crappy excel spreadsheets. 

Included are the preprocessing scripts and the resulting database, 
but NOT the original data from which the data was derived (which
is sometimes under murky copyright status). 

Each dataset is in a separate folder organized by where the data came
from, a readme (containing notes explaining the data, the schema,
etc.), the processing scripts, and the actual DB. We use the ruffus 
job-running framework 

If you are simply looking for an easy-connectivity-matrix to run
algorithm X against, this might not be your target source. My goal is
to provide scientifically-actionable-and-publishable connectivity
and metadata. 

Revisions: 

As bugs in our conversion script are identified, we will update the
data. 

As datasets are revised (corrected), we will update the data. 

Our goal is to never break your existing scientific code depending 
on this data. 

Tools
======
Preprocessing is done in python with ruffus, pandas, etc. and a cute little
ORM mapper called peewee. 


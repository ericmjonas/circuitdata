
from peewee import *



DB_NAME = "celegans_herm.db"

db = SqliteDatabase(DB_NAME)

class BaseModel(Model):
    class Meta:
        database = db # this model uses the people database


class Cells(BaseModel):
    cell_id = IntegerField(primary_key=True, db_column='cell_id')
    cell_name = CharField()
    cell_class = CharField()
    soma_pos = FloatField()
    neurotransmitters = CharField(null=True)
    role = CharField(null=True)

class Synapses(BaseModel):
    
    id = PrimaryKeyField()
    from_id = ForeignKeyField(Cells, related_name = "from", 
                              db_column="from_id")
    to_id = ForeignKeyField(Cells, related_name = "to", 
                            db_column = 'to_id')

    synapse_type = CharField()
    count = IntegerField()


def create_db():
    Cells.create_table()
    Synapses.create_table()


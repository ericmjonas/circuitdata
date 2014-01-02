
from peewee import *



DB_NAME = "mouseretina.db"

db = SqliteDatabase(DB_NAME)

class BaseModel(Model):
    class Meta:
        database = db # this model uses the people database


class Types(BaseModel):
    type_id = IntegerField(primary_key=True, 
                           db_column='type_id')
    designation = CharField(null=True)
    volgyi_type = CharField(null=True)
    macneil_type = CharField(null=True)
    coarse = CharField(null=True)
    certainty = CharField(null=True)

class Cells(BaseModel):
    cell_id = IntegerField(primary_key=True)
    type_id = ForeignKeyField(Types, related_name="cell_type", 
                              db_column='type_id')

class SomaPositions(BaseModel):
    id = PrimaryKeyField()
    cell_id = ForeignKeyField(Cells, related_name = "soma_pos", 
                              db_column='cell_id')
    x = FloatField()
    y = FloatField()
    z = FloatField()
                              
class Contacts(BaseModel):
    id = PrimaryKeyField()
    from_id = ForeignKeyField(Cells, related_name = "from", 
                              db_column="from_id")
    to_id = ForeignKeyField(Cells, related_name = "to", 
                            db_column = 'to_id')
    x = FloatField()
    y = FloatField()
    z = FloatField()
    area = FloatField()


def create_db():
    Types.create_table()
    Cells.create_table()
    SomaPositions.create_table()
    Contacts.create_table()


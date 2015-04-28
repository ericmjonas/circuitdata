
from peewee import *



DB_NAME = "mos6502.db"

db = SqliteDatabase(DB_NAME)

class BaseModel(Model):
    class Meta:
        database = db # this model uses the people database



class Wires(BaseModel):
    id = IntegerField(primary_key = True, db_column='id')
    c1c2s = IntegerField()
    gates = IntegerField()
    pullup = BooleanField()
    name = CharField(null=True)


class Transistors(BaseModel):
    name = CharField(primary_key=True, db_column='name')
    on = BooleanField()
    # pins
    gate = ForeignKeyField(Wires, related_name = 'gate')
    c1 = ForeignKeyField(Wires, related_name = 'c1')
    c2 = ForeignKeyField(Wires, related_name = 'c2')
    
    # bounding box 
    bb_x1 = FloatField()
    bb_x2 = FloatField()
    bb_y2 = FloatField()
    bb_y2 = FloatField()
    x = FloatField()
    y = FloatField()





def create_db():
    Transistors.create_table()
    Wires.create_table()

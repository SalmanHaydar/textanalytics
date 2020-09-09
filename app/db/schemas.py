from pydantic import BaseModel
from typing import List, Optional
import datetime

class InsertBase(BaseModel):

    uid : Optional[str] = None
    pgid : Optional[str] = None
    text : str
    positive : int
    negative : int
    neutral : int
    query : int
    complain : int
    appreciation : int
    feedback : int
    spam : int
    pi : int
    not_pi : int
    service_drop : int
    time : datetime.datetime
    
    class Config:
        orm_mode = True
        
class ResultList(BaseModel):
    values : List[InsertBase] = None
    
    class Config:
        orm_mode = True
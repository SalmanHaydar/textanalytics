from pydantic import BaseModel
from typing import List, Optional
import datetime

class InsertBase(BaseModel):

    uid : Optional[str] = None
    pgid : Optional[str] = None
    text : str = None
    positive : int = 0
    negative : int = 0
    neutral : int = 0
    query : int = 0
    complain : int = 0
    appreciation : int = 0
    feedback : int = 0
    spam : int = 0
    pi : int = 0
    not_pi : int = 0
    service_drop : int = 0
    time : datetime.datetime = None
    
    class Config:
        orm_mode = True
        
class ResultList(BaseModel):
    values : List[InsertBase] = None
    
    class Config:
        orm_mode = True


class Summary(BaseModel):
    uid : Optional[str] = None
    pgid : Optional[str] = None
    positive : int = 0
    negative : int = 0
    neutral : int = 0
    query : int = 0
    complain : int = 0
    appreciation : int = 0
    feedback : int = 0
    spam : int = 0
    pi : int = 0
    not_pi : int = 0
    service_drop : int = 0
    
    class Config:
        orm_mode = True
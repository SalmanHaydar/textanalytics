from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, EmailStr, Field

from fastapi import FastAPI, Body, Depends, File, Form, HTTPException, status, UploadFile, Path, Query
from fastapi.responses import HTMLResponse
from fastapi.encoders import jsonable_encoder
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

import subprocess as sp

import torch
from torch import nn
import numpy as np

from .sentModel import AttentionModel, infer_sentiment
from .queryModel import AttentionModelQuery, infer_query
from .piModel import AttentionModelPI, infer_pi

import pickle
import os
from datetime import datetime

import uvicorn

from sqlalchemy.orm import Session
from .db import crud, schemas, models
from .db.database import sessionLocal, engine

import subprocess as sp
from datetime import datetime
from dateutil import parser


oauth2_schema = OAuth2PasswordBearer(tokenUrl="token")

models.Base.metadata.create_all(bind=engine)

app = FastAPI()
print(__name__)

def get_db():
    db = sessionLocal()
    try:
        yield db
    finally:
        db.close()

#*********************************************************************************


ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

device = ("cuda:0" if torch.cuda.is_available() else "cpu")

sent_model_name = "sentiment_model_ngram_sd.pt"
sent_tokenizer_name = "tokenizer2_sent_attention_ngram.pickle"
query_model_name = "Query_model_ngram_sd.pt"
query_tokenizer_name = "tokenizer_Query_model_ngram.pickle"
pi_model_name = "PI_model_sd.pt"
pi_tokenizer_name = "tokenizer_pi_attention.pickle"

def load_model(path, mod_name=None):
    if mod_name=="sent":
        model = AttentionModel(100,100,1000)
    elif mod_name=="query":
        model = AttentionModelQuery(100,100,2000)
    elif mod_name=="pi":
        model = AttentionModelPI(100,100,2000)
        
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model

def load_tokenizer(path):
    with open(path, "rb") as f:
        t = pickle.load(f)
        return t

try:

    smod = load_model(os.path.join(ROOT_PATH,"models/"+sent_model_name),"sent")
    qmod = load_model(os.path.join(ROOT_PATH,"models/"+query_model_name),"query")
    pmod = load_model(os.path.join(ROOT_PATH,"models/"+pi_model_name),"pi")
    
    stok = load_tokenizer(os.path.join(ROOT_PATH,"models/"+sent_tokenizer_name))
    qtok = load_tokenizer(os.path.join(ROOT_PATH,"models/"+query_tokenizer_name))
    ptok = load_tokenizer(os.path.join(ROOT_PATH,"models/"+pi_tokenizer_name))
except FileNotFoundError as e:
    raise str(e)

#****************************************************************************************************


@app.get("/api/v1/analysis")
async def get_results(text:str=Query(...), uid:str=Query(None), pgid:str=Query(None)):
    payload = {"api_version": '0.0.1', 
               "DateTime": "", 
               "text": text,
               "models": {"sentiment": {"value": "", "probs": []},
                          "genCategory":{"value": "", "probs":[]},
                          "busIntention":{"value": "", "probs":[]}
                        }
               }
    
    r, p = infer_sentiment(text, stok, smod)
    # payload["sentiment"] = r
    payload["models"]["sentiment"]["value"]=r
    # payload["models"]["sentiment"]["probs"]=np.array(p[0])
    
    r, p = infer_query(text, qtok, qmod)
    # payload["genCategory"]=r
    payload["models"]["genCategory"]["value"]=r
    # payload["models"]["genCategory"]["probs"]=np.array(p[0])
    
    r, p = infer_pi(text, ptok, pmod)
    # payload["busIntention"]=r
    payload["models"]["busIntention"]["value"]=r
    # payload["models"]["busIntention"]["probs"]=np.array(p[0])
    
    payload["DateTime"] = str(datetime.now())
    
    return payload

@app.post("/api/v1/insertdata", response_model=schemas.InsertBase)
def insert_intodb(row_: schemas.InsertBase, db: Session = Depends(get_db)):
    return crud.insert_data(db, row_)

@app.get("/api/v1/getuser", response_model=List[schemas.InsertBase])
def get_user(uid:int, db: Session = Depends(get_db)):
    return crud.get_user(db, uid)

@app.get("/api/v1/gettimedata", response_model=List[schemas.InsertBase])
def get_time_data(uid: str, pgid: str, frm: str, to: str=None , db: Session = Depends(get_db)):
    return crud.get_data_by_date(db, uid, pgid, parser.parse(frm))
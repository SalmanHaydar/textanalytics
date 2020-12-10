from typing import Optional, List
from enum import Enum
from pydantic import BaseModel, EmailStr, Field

from fastapi import FastAPI, Body, Depends, File, Form, HTTPException, Request, status, UploadFile, Path, Query
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
import requests

import uvicorn

from sqlalchemy.orm import Session
from .db import crud, schemas, models
from .db.database import sessionLocal, engine
from .process import get_summary, get_VisData

import subprocess as sp
from datetime import datetime
from dateutil import parser


models.Base.metadata.create_all(bind=engine)

app = FastAPI()

print(__name__)

def get_db():
    db = sessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db2():
    db = sessionLocal()
    try:
        return db
    except:
        db.close()

def pushindb(row_: schemas.InsertBase):
    db = get_db2()
    #db.close()
    return crud.insert_data(row_, db)


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
    
    db_payload = schemas.InsertBase().dict()

    db_payload["text"]=text
    db_payload["uid"]=uid
    db_payload["pgid"]=pgid

    r, p = infer_sentiment(text, stok, smod)
    payload["models"]["sentiment"]["value"]=r
    db_payload[r.lower()] = 1
    
    r, p = infer_query(text, qtok, qmod)
    payload["models"]["genCategory"]["value"]=r
    db_payload[r.lower()] = 1
    
    r, p = infer_pi(text, ptok, pmod)
    payload["models"]["busIntention"]["value"]=r
    db_payload[r.lower()] = 1
    
    payload["DateTime"] = str(datetime.now())
    db_payload["time"] = str(datetime.now())

    if uid != None and pgid != None:
        resp = pushindb(db_payload)
    
    return payload

@app.post("/api/v1/pushdata", response_model=schemas.InsertBase)
def insert_intodb(row_: schemas.InsertBase, db: Session = Depends(get_db)):
    return crud.insert_data(row_, db)

@app.get("/api/v1/getUserDataForVisAll")
def get_user(uid:str, pgid:str, rawdata:bool = False, db: Session = Depends(get_db)):
    rows = crud.get_user(db, uid, pgid)
    if rawdata:
        return rows

    row = get_VisData(rows)
    return row

@app.get("/api/v1/getUserDataForVisTimeRange")
def get_time_data(uid: str, pgid: str, frm: str, to: str=None, rawdata:bool = False, db: Session = Depends(get_db)):
    rows = crud.get_data_by_date(db, uid, pgid, parser.parse(frm), parser.parse(to))
    if rawdata:
        return rows

    row = get_VisData(rows)

    return row

@app.get("/api/v1/getPageDataForVisAll")
def get_time_data(pgid: str, rawdata:bool = False, db: Session = Depends(get_db)):
    rows = crud.get_page_data_all(db, pgid)
    if rawdata:
        return rows

    row = get_VisData(rows)

    return row

@app.get("/api/v1/getPageDataForVisTimeRange")
def get_time_data(pgid: str, frm: str, to: str=None, rawdata:bool = False, db: Session = Depends(get_db)):
    rows = crud.get_page_data_by_date(db, pgid, parser.parse(frm), parser.parse(to))
    if rawdata:
        return rows

    row = get_VisData(rows)

    return row


@app.get("/api/v1/getPages")
def get_page_list(db: Session = Depends(get_db)):
    rows = crud.get_all_pages(db)
    
    return rows

@app.get("/api/v1/getUsers")
def get_user_list(pgid: str, db: Session = Depends(get_db)):
    rows = crud.get_all_userid(db, pgid)
    
    return rows

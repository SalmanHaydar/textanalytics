from sqlalchemy.orm import Session
from dateutil import parser
from datetime import datetime

from . import models, schemas


def get_user(db: Session, user_id: str, pg_id: str):
    return db.query(models.Analytics).filter(models.Analytics.pgid == pg_id).filter(models.Analytics.uid == user_id).all()

def insert_data(row: schemas.InsertBase, db: Session):
    if type(row) == schemas.InsertBase:
        row_ = models.Analytics(**row.dict())
    else:
        row_ = models.Analytics(**row)
    db.add(row_)
    db.commit()
    db.refresh(row_)
    return row_

def get_data_by_date(db: Session, uid: str, pgid: str, frm: datetime, to: datetime):
    return db.query(models.Analytics).filter(models.Analytics.pgid == pgid).filter(models.Analytics.uid == uid).filter(models.Analytics.time >= frm).filter(to >= models.Analytics.time).all()

def get_page_data_all(db: Session, pgid: str):
    return db.query(models.Analytics).filter(models.Analytics.pgid == pgid).all()

def get_page_data_by_date(db: Session, pgid: str, frm: datetime, to: datetime):
    return db.query(models.Analytics).filter(models.Analytics.pgid == pgid).filter(models.Analytics.time >= frm).filter(to >= models.Analytics.time).all()

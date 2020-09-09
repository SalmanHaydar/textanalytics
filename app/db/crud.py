from sqlalchemy.orm import Session
from dateutil import parser
from datetime import datetime

from . import models, schemas


def get_user(db: Session, user_id: str):
    return db.query(models.Analytics).filter(models.Analytics.uid == user_id).all()

def insert_data(db: Session, row: schemas.InsertBase):
    row_ = models.Analytics(**row.dict())
    db.add(row_)
    db.commit()
    db.refresh(row_)
    return row_

def get_data_by_date(db: Session, uid: str, pgid: str, frm: datetime=None, to: datetime=None):
    return db.query(models.Analytics).filter(models.Analytics.uid == uid and models.Analytics.pgid == pgid).filter(models.Analytics.time > frm).all()

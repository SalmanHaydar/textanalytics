from sqlalchemy import Column, Text, DateTime, INT, VARCHAR, Integer, text
from sqlalchemy.orm import relationship

from .database import Base


class Analytics(Base):
    __tablename__ = "textAnalytics"
    
    id = Column(Integer , nullable=False, index=True, primary_key=True, autoincrement=True)
    uid = Column(VARCHAR(85), default='DEFAULT_id', nullable=False)
    pgid = Column(VARCHAR(50), default='DEFAULT_pgid', nullable=False)
    text = Column(Text(1000), nullable=False)
    positive = Column(Integer, default=0, nullable=False)
    negative = Column(Integer, default=0, nullable=False)
    neutral = Column(Integer, default=0, nullable=False)
    query = Column(Integer, default=0, nullable=False)
    complain = Column(Integer, default=0, nullable=False)
    appreciation = Column(Integer, default=0, nullable=False)
    feedback = Column(Integer, default=0, nullable=False)
    spam = Column(Integer, default=0, nullable=False)
    pi = Column(Integer, default=0, nullable=False)
    not_pi = Column(Integer, default=0, nullable=False)
    service_drop = Column(Integer, default=0, nullable=False)
    time = Column(DateTime, nullable=False)
    
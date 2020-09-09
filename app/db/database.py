from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

SQL_DB_URL = 'mysql://root:genexdb2020@182.160.104.220:3306/Analytics'

engine = create_engine(SQL_DB_URL)

sessionLocal = sessionmaker(autoflush=False, bind=engine)

Base = declarative_base()

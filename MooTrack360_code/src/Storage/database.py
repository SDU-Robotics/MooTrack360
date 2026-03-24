from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .cow_instance import Base

def init_db(db_url="sqlite:///cows.db",):
    """
    Initializes the database.
    db_url can be set to your database of choice, e.g., 'postgresql://user:pass@localhost/dbname'
    """
    engine = create_engine(db_url, echo=False, connect_args={"check_same_thread": False})
    # if reset:
        # Base.metadata.drop_all(engine)  # Drops any existing tables
    # 
    Base.metadata.create_all(engine)    # Creates tables if they don't exist
    return engine

# Create a session factory
engine = init_db()
Session = sessionmaker(bind=engine)

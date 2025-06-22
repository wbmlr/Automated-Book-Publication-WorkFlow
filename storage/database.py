from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy import Column, Integer, String, Text, LargeBinary, DateTime
from sqlalchemy.sql import func
from urllib.parse import urlparse

from config import DATABASE_URL

# --- Database Creation Logic ---
def create_database_if_not_exists(url: str):
    """
    Connects to the default 'postgres' database to create the target
    database if it doesn't already exist.
    """
    try:
        parsed_url = urlparse(url)
        db_name = parsed_url.path.lstrip('/')
        port = parsed_url.port or 5432
        postgres_db_url = f"postgresql://{parsed_url.username}:{parsed_url.password}@{parsed_url.hostname}:{port}/postgres"
        
        engine = create_engine(postgres_db_url)
        with engine.connect() as conn:
            # Check if the database exists
            result = conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname='{db_name}'"))
            db_exists = result.scalar() == 1

            if not db_exists:
                print(f"Database '{db_name}' not found. Creating it...")
                # FIX: Commit the current transaction before changing isolation level.
                conn.commit() 
                
                # Now that the previous transaction is closed, we can execute CREATE DATABASE.
                conn.execution_options(isolation_level="AUTOCOMMIT").execute(text(f"CREATE DATABASE {db_name}"))
                print(f"Database '{db_name}' created successfully.")
            else:
                print(f"Database '{db_name}' already exists.")
    except OperationalError as e:
        print(f"FATAL: Could not connect to PostgreSQL server. Is it running? Error: {e}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred during DB setup: {e}")
        exit()

# ... (the rest of the file remains exactly the same) ...

# --- Main Application DB Setup ---
# Call the creation logic before setting up the main engine
create_database_if_not_exists(DATABASE_URL)

# Now, connect to the application's database (which is guaranteed to exist)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Model Definition ---
class ScrapedContent(Base):
    __tablename__ = "scraped_content"
    id = Column(Integer, primary_key=True, index=True)
    url = Column(String, unique=True, index=True, nullable=False)
    raw_text = Column(Text, nullable=False)
    screenshot = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

def init_db():
    """Creates the tables in the database."""
    try:
        # Check if table exists before creating
        inspector = inspect(engine)
        if not inspector.has_table(ScrapedContent.__tablename__):
            print(f"Table '{ScrapedContent.__tablename__}' not found. Creating it...")
            Base.metadata.create_all(bind=engine)
            print("Table created.")
        else:
            print(f"Table '{ScrapedContent.__tablename__}' already exists.")
    except Exception as e:
        print(f"Could not create table. Error: {e}")
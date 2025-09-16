from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

engine = create_engine(settings.database_url, future=True)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


def get_db() -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations."""

    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

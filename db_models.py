from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import sessionmaker, declarative_base

# Database Setup
#TODO: Make more secure
DATABASE_URL = 'sqlite:///jobs_data.db'
engine = create_engine(DATABASE_URL)
Base = declarative_base()

# LinearJob Model Definition
class LinearJob(Base):
    __tablename__ = 'linear_jobs'
    job_id = Column(String, primary_key=True) 
    uuid = Column(String) 
    status = Column(String, default='UNPROCESSED')

    def __init__(self, job_id, uuid, status='UNPROCESSED'):
        self.job_id = job_id
        self.uuid = uuid
        self.status = status

    def __repr__(self):
        return f"<LinearJob(job_id='{self.job_id}', uuid='{self.uuid}', status='{self.status}')>"

# Create tables
Base.metadata.create_all(engine)

# Session Factory
SessionLocal = sessionmaker(bind=engine)


def uuid_from_job(job_id: str) -> str | None:
    """Retrieves the uuid for a given job_id."""
    with SessionLocal() as session:
        job = session.query(LinearJob).filter(LinearJob.job_id == job_id).first()
        return job.uuid if job else None

def update_status(target_uuid: str) -> bool:
    """
    Checks the job status and updates it from "UNPROCESSED" to "FINISHED" using UUID.

    Args:
        target_uuid: The uuid of the job to update.

    Returns:
        True if the status was updated, False otherwise.
    """
    with SessionLocal() as session:
        job = session.query(LinearJob).filter(LinearJob.uuid == target_uuid).first()

        if not job:
            print(f"Job with UUID '{target_uuid}' not found.")
            return False

        if job.status == 'UNPROCESSED':
            job.status = 'FINISHED'
            session.commit()
            print(f"Status for UUID {target_uuid} (Job ID: {job.job_id}) updated to FINISHED.")
            return True
        else:
            print(f"Status for UUID {target_uuid} is already '{job.status}'. No update needed.")
            return False


def processing_status(target_uuid: str) -> str | None:
    """
    Returns the current status ("UNPROCESSED" or "FINISHED") of a job using UUID.

    Args:
        target_uuid: The uuid to check.

    Returns:
        The status as a string, or None if the uuid is not found.
    """
    with SessionLocal() as session:
        job = session.query(LinearJob).filter(LinearJob.uuid == target_uuid).first()
        
        if job:
            return job.status
        else:
            return None

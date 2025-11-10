from pathlib import Path

import numpy as np
import arviz as az
from sturdystats.model import Model, 
from sturdystats.job import Job

import sqlite3
from sqlalchemy import create_engine, Column, String
from sqlalchemy.orm import sessionmaker, declarative_base

import db_models

dpath = "/Users/kian/ML/numix/src/regdata/"

#should be called when kicking off function IF check_status returns "UNSTARTED"
def preprocessing(model: Model, X, Y,
                  label_names,
                  feature_names, job):
    #make empty inference data obj
    inference_data = az.InferenceData(posterior = {})

    #store all preprocessing info
    inference_data = inference_data.assign_coords({"Q": label_names, "dim": feature_names})
    inference_data.attrs["model_type"] = model.model_type
    inference_data.attrs["label_names"] = list(label_names)
    inference_data.attrs["feature_names"] = list(feature_names)
    inference_data.attrs["job"] = job
    inference_data.attrs["X"] = X
    inference_data.attrs["Y"] = Y
    
    
    model.inference_data = inference_data

    #get uuid from job_id
    uuid = db_models.uuid_from_job(job.job_id)
    if uuid is None:
        #TODO throw error
        return 0
    
    #proof of concept for data management (LEARN HOW TO SAVE TO SERVER PROPERLY)
    model.to_disk("regdata/" + uuid + ".ss")


#Returns the status of a job if the model was kicked off earlier
#Returns "FINISHED" if the model has already been through postprocessing
#Does postprocessing if the job has completed but hasn't "FINISHED"
#Returns "UNSTARTED" if the model has never been kicked off before
def check_status(uuid):

    #check that preprocessing was done properly
    if os.path.exists("regdata/" + uuid + ".ss"):
        preprocessed = Model.from_disk("regdata/" + uuid + ".ss")

        
        #Check if the job was already post-processed
        if db_models.processing_status(uuid) == "FINISHED":
            return "FINISHED"
        
        job = inference_data.attrs["job"]
        status = job.get_status()

        #see if job has finished loading
        if status["status"] in ["SUCCEEDED"]:
            postprocessing(uuid)
            returns "FINISHED"
            
        return status
        #caller function should take this output and know
        #if FINISHED: load page
        #if status["status"] RUNNING: show loading screen/message saying its still running
        #if status["status"] FAILED: show some explanation of error
        #if status["status"] CANCELLED: was cancelled by user


        
    else:
        return "UNSTARTED"
        #Show that this model has never kicked off before

#Takes stored preprocessing metadata and appends the metadata appropriately to the finished job's getTrace()
def postprocessing(uuid):
    
    preprocessed = Model.from_disk("regdata/" + uuid + ".ss")

    job = inference_data.attrs["job"]
    inference_data = job.getTrace()

    X = preprocessed.inference_data.attrs["X"]
    Y = preprocessed.inference_data.attrs["Y"]
    
    inference_data.assign_coords({"Q": preprocessed.inference_data.attrs["label_names"], "dim": preprocessed.inference_data.attrs["feature_names"]})
    inference_data.attrs["model_type"] = preprocessed.inference_data.attrs["model_type"]
    inference_data.attrs["label_names"] = preprocessed.inference_data.attrs["label_names"]
    inference_data.attrs["feature_names"] = preprocessed.inference_data.attrs["feature_names"]
    
    preprocessed.inference_data = inference_data

    Model._append_data(preprocessed.inference_data, X, Y)
    Model.inference_data.add_groups(
            posterior_predictive=preprocessed.sample_posterior_predictive(X))

    db_models.update_status(uuid)
    #TODO: Put some error checks to return an error message if something goes wrong



    
# Define the base class for declarative class definitions
Base = declarative_base()
#class for SQLite
class LinearJob(Base):
    # Name of the table in the database
    __tablename__ = 'linear_jobs'

    job_id = Column(String, primary_key=True)
    uuid = Column(String)
    status = Column(String)

    def __init__(self, job_id, uuid, status):
        self.job_id = job_id
        self.uuid = uuid
        self.status = status

    def __repr__(self):
        return f"<LinearJob(job_id='{self.job_id}', uuid='{self.uuid}', status='{self.status}')>"






#TODO: make this engine more secure
engine = create_engine('sqlite:///jobs_data.db')
SessionLocal = sessionmaker(bind=engine)

def uuid_from_job(job_id: str) -> str | None:
    """
    Retrieves the uuid for a given job_id using SQLAlchemy.

    Args:
        job_id: The job_id to look up.

    Returns:
        The uuid as a string, or None if the job_id is not found.
    """
    with SessionLocal() as session:
        # Query the LinearJob model for the record with the matching job_id
        job = session.query(LinearJob).filter(LinearJob.job_id == job_id).first()
        
        if job:
            return job.uuid
        else:
            return None

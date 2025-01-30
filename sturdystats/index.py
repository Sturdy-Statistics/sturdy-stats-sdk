import requests
import time
import os
import json
from sturdystats.job import Job

import srsly                           # to decode output
from more_itertools import chunked     # to batch data for API calls
from tenacity import (
    retry,
    stop_after_attempt,
) 
from spacy.tokens import Doc, Span, Token, DocBin



# for type checking
from typing import Optional, Iterable, Dict
from requests.models import Response


class Index:

    ## TODO support id based loading as well if already exists
    def __init__(
            self,
            API_key: Optional[str] = None,
            name: Optional[str] = None,
            id: Optional[str] = None,
            _base_url: Optional[str] = None,
            verbose: bool = True
    ):

        self.API_key = API_key or os.environ["STURDY_STATS_API_KEY"]
        self.base_url = _base_url or "https://sturdystatistics.com/api/v1/text/index"

        self.name = name
        self.id = id
        self.verbose = verbose

        if (self.name is None) and (self.id is None):
            raise ValueError("Must provide either an index_name or an index_id.")
        if (self.name is not None) and (self.id is not None):
            raise ValueError("Cannot provide both an index_name and an index_id.")

        status = self._get_status(index_name=self.name, index_id = self.id)
        if status is None:
            self.id = self._create(self.name)
            self._print(f"""Created new index with id="{self.id}".""")
        else:
            self.id = status["id"]
            self._print(f"""Found an existing index with id="{self.id}".""")
        self.pandata = None


    def _print(self, *msg):
        if self.verbose: print(*msg)


    def _job_base_url(self) -> str:
        return self.base_url.replace("text/index", "job")

    def _check_status(self, info: Response) -> None:
        if info.status_code != 200:
            raise requests.HTTPError(info.content)

    #@retry(stop=(stop_after_attempt(3)))
    def _post(self, url: str, params: Dict) -> Response:
        payload = {**params}
        res = requests.post(self.base_url + url, json=payload, headers={"x-api-key": self.API_key})
        self._check_status(res)
        return res

    def _get(self, url: str, params: Dict) -> Response:
        params = {**params}
        res = requests.get(self.base_url + url , params=params, headers={"x-api-key": self.API_key})
        self._check_status(res)
        return res



    def _create(self, index_name: str):
        """Creates a new index. An index is the core data structure for
    storing data.  Once the index is trained, an index may also be
    used to search, query, and analyze data. If an index with the
    provided name already exists, no index will be created and the
    metadata of that index will be returned.

    https://sturdystatistics.com/api/documentation#tag/apitextv1/operation/createIndex

    """

        # Create a new index associated with this API key.  Equivalent to:
        #
        # curl -X POST https://sturdystatistics.com/api/text/v1/index \
        #   -H "Content-Type: application/json" \
        #   -d '{
        #      "api_key": "API_KEY",
        #      "name": "INDEX_NAME"
        #    }'

        info = self._post("", dict(name=index_name))
        index_id = info.json()["id"]
        return index_id



    def _get_status_by_name(self, index_name: str):

        # List all indices associated with this API key.  Equivalent to:
        #
        # curl -X GET 'https://sturdystatistics.com/api/text/v1/index?api_key=API_KEY'
        #
        # https://sturdystatistics.com/api/documentation#tag/apitextv1/operation/listIndicies

        info = self._get("", dict())

        # find matches by name
        matches = [ i for i in info.json() if i["name"] == index_name ]
        if (0 == len(matches)):
            return None
        assert(1 == len(matches))
        return matches[0]



    def _get_status_by_id(self, index_id: str):

        # curl -X GET 'https://sturdystatistics.com/api/text/v1/index/{index_id}?api_key=API_KEY'

        info = self._get(f"/{index_id}", dict())
        status = info.json()
        return status



    def _get_status(self,
                   index_name: Optional[str] = None,
                   index_id: Optional[str] = None):
        """Look up an index by name or ID and return all metadata
    associated with the index.

    https://sturdystatistics.com/api/documentation#tag/apitextv1/operation/getSingleIndexInfo

    """

        if (index_name is None) and (index_id is None):
            raise ValueError("Must provide either an index_name or an index_id.")
        if (index_name is not None) and (index_id is not None):
            raise ValueError("Cannot provide both an index_name and an index_id.")
        if index_id is not None:
            # look up by index_id:
            return self._get_status_by_id(index_id)
        # look up by name:
        return self._get_status_by_name(index_name)

    def get_status(self) -> dict:
        if self.id is not None:
            return self._get_status(index_id=self.id)
        else:
            return self._get_status(index_name=self.name)

    def commit(self, wait: bool = True):
        """
        """
        self._print(f"""committing changes to index "{self.id}"...""")
        # Commit changes from the staging index to the permanent index.  Equivalent to:
        #
        # curl -X POST https://sturdystatistics.com/api/text/v1/index/{index_id}/doc/commit \
        #   -H "Content-Type: application/json" \
        #   -d '{
        #      "api_key": "API_KEY",
        #    }'
        info = self._post(f"/{self.id}/doc/commit", dict())
        job_id = info.json()["job_id"]
        job = Job(self.API_key, job_id, 20, _base_url=self._job_base_url())
        if not wait:
            return job
        return job.wait()

    def unstage(self, wait: bool = True):
        """
        """
        self._print(f"""unstaging changes to index "{self.id}"...""") 
        # Commit changes from the staging index to the permanent index.  Equivalent to:
        #
        # curl -X POST https://sturdystatistics.com/api/text/v1/index/{index_id}/doc/commit \
        #   -H "Content-Type: application/json" \
        #   -d '{
        #      "api_key": "API_KEY",
        #    }'
        info = self._post(f"/{self.id}/doc/unstage", dict())
        job_id = info.json()["job_id"]
        job = Job(self.API_key, job_id, 5, _base_url=self._job_base_url())
        if not wait:
            return job
        return job.wait()


    def _upload_batch(self, records: Iterable[Dict], save = "true"):
        if len(records) > 1000:
            raise RuntimeError(f"""The maximum batch size is 1000 documents.""")
        info = self._post(f"/{self.id}/doc", dict(docs=records, save=save))
        job_id = info.json()["job_id"]
        job = Job(self.API_key, job_id, 5, _base_url=self._job_base_url())
        return job.wait()


    def upload(self,
              records: Iterable[Dict],
              batch_size: int = 1000,
              commit: bool = True):
        """Uploads documents to the index and commit them for
    permanent storage.  Documents are processed by the AI model if the
    index has been trained.

    Documents are provided as a list of dictionaries. The content of
    each document must be plain text and is provided under the
    required field doc.  You may provide a unique document identifier
    under the optional field doc_id. If no doc_id is provided, we will
    create an identifier by hashing the contents of the
    document. Documents can be updated via an upsert mechanism that
    matches on doc_id. If doc_id is not provided and two docs have
    identical content, the most recently uploaded document will upsert
    the previously uploaded document.

    This is a locking operation. A client cannot call upload, train or
    commit while an upload is already in progress. Consequently, the
    operation is more efficient with batches of documents. The API
    supports a batch size of up to 1000 documents at a time. The larger
    the batch size, the more efficient the upload.

    https://sturdystatistics.com/api/documentation#tag/apitextv1/operation/writeDocs

    """

        status = self.get_status()
        if "untrained" == status["state"]:
            self._print("Uploading data to UNTRAINED index for training.")
        elif "ready" == status["state"]:
            self._print("Uploading data to TRAINED index for prediction.")
        else:
            raise RuntimeError(f"""Unknown status "{status['state']}" for index "{self.name}".""")
        results = []
        # Upload docs to the staging index.  Equivalent to:
        #
        # curl -X POST https://sturdystatistics.com/api/text/v1/index/{index_id}/doc \
        #   -H "Content-Type: application/json" \
        #   -d '{
        #      "api_key": "API_KEY",
        #      "docs": JSON_DOC_DATA
        #    }'

        self._print("uploading data to index...")
        batch = []
        maxsize = 1e7 - 1e6
        cursize = 0
        docIsNull = False
        for i, doc in enumerate(records):
            if not docIsNull and len( (doc.get("doc", "") or "").strip()) == 0:
                print(" Warning: field `doc` is empty. Empty documents are allowed but only data stored under the field `doc` will have its content indexed")
                docIsNull = True

            docsize = len(json.dumps(doc).encode("utf-8"))
            if docsize > maxsize:
                raise RuntimeError(f"""Record number {i} is {docsize} bytes. A document cannot be larger than {maxsize} bytes""")
            if cursize + docsize > maxsize or len(batch) >= batch_size:
                info = self._upload_batch(batch)
                results.extend(info["result"]["results"])
                batch = []
                cursize = 0
                self._print(f"""    upload status: record no {i}""")
            batch.append(doc)
            cursize += docsize

        if len(batch) > 0:
            info = self._upload_batch(batch)
            results.extend(info["result"]["results"])
        if commit: self.commit()
        return results


    def train(self, params: Dict = dict(), fast: bool = False, force: bool = False, wait: bool = True):
        """Trains an AI model on all documents in the production
    index. Once an index has been trained, documents are queryable,
    and the model automatically processes subsequently uploaded
    documents.

    The AI model identifies thematic information in documents, permitting
    semantic indexing and semantic search. It also enables quantitative
    analysis of, e.g., topic trends.

    The AI model may optionally be supervised using metadata present in the
    index. Thematic decomposition of the data is not unique; supervision
    guides the model and aligns the identified topics to your intended
    application. Supervision also allows the model to make predictions.

    Data for supervision may be supplied explicitly using the
    label_field_names parameter. Metadata field names listed in this
    parameter must each store data in a ternary true/false/unknown format.
    For convenience, supervision data may also be supplied in a sparse "tag"
    format using the tag_field_names parameter. Metadata field names listed
    in this parameter must contain a list of labels for each document. The
    document is considered "true" for each label listed; it is implicitly
    considered "false" for each label not listed. Consequently, the "tag"
    format does not allow for unknown labels. Any combination of
    label_field_names and tag_field_names may be supplied.

    https://sturdystatistics.com/api/documentation#tag/apitextv1/operation/trainIndex

    """

        status = self.get_status()
        if ("untrained" != status["state"]) and not force:
            self._print(f"index {self.name} is already trained.")
            return status

        if fast:
            params["K"] = params.get("K", 96)
            params["burn_in"] = params.get("burn_in", 1000)
            params["model_args"] = " MCMC/sample_a_start=100000 " + params.get("model_args", "")

        # Issue a training command to the index.  Equivalent to:
        #
        # curl -X POST https://sturdystatistics.com/api/text/v1/index/{index_id}/train \
        #   -H "Content-Type: application/json" \
        #   -d '{
        #      "api_key": "API_KEY",
        #      PARAMS
        #    }'

        info = self._post(f"/{self.id}/train", params)
        job_id = info.json()["job_id"]
        job = Job(self.API_key, job_id, 30, _base_url=self._job_base_url())
        if wait:
            return job.wait()
        else:
            return job



    def predict(self, records: Iterable[Dict], batch_size: int = 1000):
        """"Predict" function analogous to sklearn or keras: accepts
    a batch of documents and returns their corresponding predictions.

    Performs an upload operation with `save=false` and without a commit step.
    This function does not mutate the index in any way.

    https://sturdystatistics.com/api/documentation#tag/apitextv1/operation/writeDocs

    """

        status = self.get_status()

        if "ready" != status["state"]:
            raise RuntimeError(f"""Cannot run predictions on index "{self.name}" with state {status["state"]}.""")


        results = []

        # Upload docs to the staging index.  Equivalent to:
        #
        # curl -X POST https://sturdystatistics.com/api/text/v1/index/{index_id}/doc \
        #   -H "Content-Type: application/json" \
        #   -d '{
        #      "api_key": "API_KEY",
        #      "save": "false",
        #      "docs": JSON_DOC_DATA
        #    }'

        self._print("running predictions...")
        for i, batch in enumerate(chunked(records, batch_size)):
            info = self._upload_batch(batch, save="false")
            results.extend(info["result"]['results'])
            self._print(f"""    upload batch {1+i:4d}: response {str(info)}""")
            self._print("...done")

            # no commit needed since this makes no change to the index

        return results

    def query(
        self,
        search_query: Optional[str] = None,
        topic_id: Optional[int] = None,
        topic_group_id: Optional[int] = None,
        filters: str = "",
        offset: int = 0,
        limit: int = 20,
        sort_by: str = "relevance",
        ascending: bool = False,
        context: int = 0,
        max_excerpts_per_doc: int = 1,
        override_args: dict = dict()
    ):
        params = dict(
            offset=offset,
            limit=limit,
            sort_by=sort_by,
            ascending=ascending,
            filters=filters,
            context=context,
            max_excerpts_per_doc=max_excerpts_per_doc
        )
        if search_query is not None:
            params["query"] = search_query
        if topic_id is not None:
            params['topic_ids'] = topic_id
        if topic_group_id is not None:
            params["topic_group_id"] = topic_group_id
        params = {**params, **override_args}
        res = self._get(f"/{self.id}/doc", params)
        return res.json()

    def getDocs(
        self,
        doc_ids: list[str],
        search_query: Optional[str] = None,
        topic_id: Optional[int] = None,
        topic_group_id: Optional[int] = None,
        context: int = 0,
        override_args: dict = dict()
    ):
        assert len(doc_ids) > 0
        params = dict(context=context)
        if search_query is not None:
            params["query"] = search_query
        if topic_id is not None:
            params['topic_ids'] = topic_id
        if topic_group_id is not None:
            params["topic_group_id"] = topic_group_id
        params = {**params, **override_args}
        joined = ",".join(doc_ids)
        return self._get(f"/{self.id}/doc/{joined}", params).json()

    def getDocsBinary(
        self,
        doc_ids: list[str],
    ):
        assert len(doc_ids) > 0
        joined = ",".join(doc_ids)
        docbin = DocBin().from_bytes(self._get(f"/{self.id}/doc/binary/{joined}", dict()).content)
        pandata: dict = self.getPandata() # type: ignore
        for tok, name in zip([Token, Span, Doc], ["token", "span", "doc"]):
            for ext in pandata.get(name+"_exts", []): 
                if not tok.has_extension(ext["name"]): tok.set_extension(**ext)
        return docbin


    def getPandata(
        self,
    ):
        if self.pandata is None:
            self.pandata = srsly.msgpack_loads(self._get(f"/{self.id}/pandata", dict()).content)
        return self.pandata

    def queryMeta(self, query: str):
        return srsly.msgpack_loads(self._get(f"/{self.id}/doc/meta", dict(q=query)).content)
    
    def annotate(self):
        self._post(f"/{self.id}/annotate", dict())
        while True:
            res = self.get_status()
            if res["state"] == "ready":
                break
            time.sleep(3)

    def clone(self, new_name):
        info = self._post(f"/{self.id}/clone", dict(new_name=new_name))
        job_id = info.json()["job_id"]
        job = Job(self.API_key, job_id, 20, _base_url=self._job_base_url())
        return job.wait()

    def delete(self, force: bool):
        if not force:
            print("Are you sure you want to delete this index? There is no going back")
            return
        return self._post(f"/{self.id}/delete", dict())

    def topicSearch(
        self,
        query: str = "",
        filters: str = "",
        doc_ids: list[str] = [],
        limit: int = 100
    ):
        params = dict(
            query=query,
            filters=filters,
            doc_ids=doc_ids,
            limit=limit,
        )
        res = self._get(f"/{self.id}/topic/search", params)
        return res.json()


    def topicDiff(
        self,
        q1: str,
        q2: str = "",
        limit: int = 50,
        cutoff: float = 1.0,
        min_confidence: float = 95,
        search_query1: str = "",
        search_query2: str = "",
        override_args: dict = dict()
    ):
        params = dict(
            q1=q1,
            limit=limit,
            cutoff=cutoff,
            min_confidence=min_confidence,
            search_query1=search_query1,
            search_query2=search_query2
        )
        if len(q2.strip()) > 0:
            params["q2"] = q2
        params = {**params, **override_args}
        res = self._get(f"/{self.id}/topic/diff", params)
        return res.json()

    def listJobs(
        self,
        status: str= "RUNNING",
        job_name: Optional[str] = None,
        only_current_index: bool = True,
    ):
        assert status in [None, "", "RUNNING", "FAILED", "SUCCEEDED", "PENDING"]
        assert job_name in [None, "", "trainIndex", "commitIndex", "unstageIndex", "writeDocs"]
        params = dict()
        if only_current_index:
            params["index_id"] = self.id
        if status is not None and status.strip() != "":
            params["status"] = status
        if job_name is not None and job_name.strip() != "":
            params["job_name"] = job_name

        job = Job(self.API_key, "", 1, _base_url=self._job_base_url())
        res = job._get("", params)
        return res.json()

    def listIndices(
        self,
        name_filter: Optional[str] = None,
        state_filter: Optional[str] = None,
    ):
        res = self._get("", dict()).json()
        results = []
        for r in res:
            if name_filter is not None and name_filter not in r["name"]:
                continue
            if state_filter is not None and state_filter != r["state"]:
                continue
            results.append(r)
        return results

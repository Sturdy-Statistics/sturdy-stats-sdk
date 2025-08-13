from __future__ import annotations

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
    wait_exponential
) 
import pandas as pd




# for type checking
from typing import Literal, Optional, Iterable, Dict, overload
from requests.models import Response

class Index:
    def __init__(
            self,
            API_key: Optional[str] = os.environ.get("STURDY_STATS_API_KEY"),
            name: Optional[str] = None,
            id: Optional[str] = None,
            _base_url: Optional[str] = None,
            verbose: bool = True,
    ):

        self.API_key = API_key or ""
        self.base_url = _base_url or "https://api.sturdystatistics.com/api/v1/text/index"

        if (name is None) and (id is None):
            raise ValueError("Must provide either an index_name or an index_id.")
        if (name is not None) and (id is not None):
            raise ValueError("Cannot provide both an index_name and an index_id.")


        if id is not None: info = self._get_status(id)
        elif name is not None: info = self._create(name)
        else: raise ValueError("Must provide an index_name XOR an index_id")

        self.id = info["id"]
        self.name = info["name"]

        self.verbose = verbose
        self.pandata = None



    def _print(self, *msg):
        if self.verbose: print(*msg)


    def _job_base_url(self) -> str:
        return self.base_url.replace("text/index", "job")

    def _check_status(self, info: Response) -> None:
        if info.status_code != 200:
            try: content = info.json()
            except: 
                try: content = info.content.decode() 
                except: content = info.content
            raise requests.HTTPError(content)

    def _post(self, url: str, params: Dict) -> Response:
        if self.API_key is None:
            raise ValueError("All POST requests (index creation, data upload, model training) require an API Key. Visit https://sturdystatistics.com/docs/web/api-key-creation.html to create your free API key today.")
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
        index_info = info.json()
        return index_info 

    def _get_status(self,
                   index_id: str
                    ) -> dict:
        info = self._get(f"/{index_id}", dict())
        return info.json()

    def get_status(self,) -> dict:
        """Look up an index by name or ID and return all metadata
    associated with the index.

    https://sturdystatistics.com/api/documentation#tag/apitextv1/operation/getSingleIndexInfo

    """
        return self._get_status(self.id) 

    @overload
    def commit(self, wait: Literal[True] = True) -> dict: ...
    @overload
    def commit(self, wait: Literal[False] = False) -> Job: ...
    def commit(self, wait: bool = True) -> Job | dict:
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
        job = Job(self.API_key, job_id, 1, _base_url=self._job_base_url())
        if not wait:
            return job
        return job.wait()

    @overload
    def unstage(self, wait: Literal[True] = True) -> dict: ...
    @overload
    def unstage(self, wait: Literal[False] = False) -> Job: ...
    def unstage(self, wait: bool = True) -> Job | dict:
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
        job = Job(self.API_key, job_id, 1, _base_url=self._job_base_url())
        if not wait:
            return job
        return job.wait()

    @overload
    def updateIntegrationDocs(self, wait: Literal[True] = True) -> dict: ...
    @overload
    def updateIntegrationDocs(self, wait: Literal[False] = False) -> Job: ...
    def updateIntegrationDocs(self, wait: bool = True) -> Job | dict:
        """
        """
        self._print(f"""updating index "{self.id}"...""") 
        # Load the latest data from all existing integrations applied to an index
        #
        # curl -X POST https://sturdystatistics.com/api/text/v1/index/{index_id}/doc/integration/update
        info = self._post(f"/{self.id}/doc/integration/update", dict())
        job_id = info.json()["job_id"]
        job = Job(self.API_key, job_id, 1, _base_url=self._job_base_url())
        if not wait:
            return job
        return job.wait()



    def _upload_batch(self, records: Iterable[Dict], save = "true"):
        if len(records) > 1000:
            raise RuntimeError(f"""The maximum batch size is 1000 documents.""")
        info = self._post(f"/{self.id}/doc", dict(docs=records, save=save))
        job_id = info.json()["job_id"]
        job = Job(self.API_key, job_id, 1, _base_url=self._job_base_url())
        return job.wait()


    def upload(self,
              records: Iterable[Dict],
              batch_size: int = 1000,
              commit: bool = True) -> list[dict]:
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
                self._print(" Warning: field `doc` is empty. Empty documents are allowed but only data stored under the field `doc` will have its content indexed")
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

    def deleteDocs(
        self,
        doc_ids: list[str],
        override_args: dict = dict()
    ) -> dict:
        assert len(doc_ids) > 0
        params = dict()
        params = {**params, **override_args}
        joined = ",".join(doc_ids)
        return self._post(f"/{self.id}/doc/delete/{joined}", params).json()

    def ingestIntegration(self,
        engine: Literal[
            "academic_search", 
            "hackernews_comments", 
            "hackernews_story", 
            "earnings_calls",
            "author_cn", 
            "news_date_split", 
            "google", 
            "google_news", 
            "reddit", 
            "cn_all",
            "apple_app_store_reviews", 
            "walmart_product_reviews",
            "home_depot_product_reviews",
        ],
        query: str,
        start_date: str | None = None, 
        end_date: str | None = None,
        args: dict = dict(),
        commit: bool = True,
        wait: bool = True,
    ) -> Job | dict:
        assert engine in [
                "earnings_calls", "hackernews_comments", "hackernews_story", 
                "academic_search", "author_cn", "news_date_split", 
                "google", "google_news", "reddit", "cn_all",
                "apple_app_store_reviews", "walmart_product_reviews",
                "home_depot_product_reviews",
                          ] 
        params = dict(q=query, engine=engine, commit=commit) 
        if start_date is not None: params["start_date"] = start_date
        if end_date is not None: params["end_date"] = end_date 
        params = params | args
        self._print("uploading data to index...")
        info = self._post(f"/{self.id}/doc/integration", params)
        job_id = info.json()["job_id"]
        job = Job(self.API_key, job_id, 1, _base_url=self._job_base_url())
        if not wait: return job
        res = job.wait()
        return res



    @overload
    def train(self, params: Dict = dict(), fast: bool = False, force: bool = False, wait: Literal[True] = True) -> dict: ...
    @overload
    def train(self, params: Dict = dict(), fast: bool = False, force: bool = False, wait: Literal[False] = False) -> Job: ...
    def train(self, params: Dict = dict(), fast: bool = False, force: bool = False, wait: bool = True) -> Job | dict:
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

        job_params = {**params}
        poll = 5
        if fast:
            job_params["K"] = job_params.get("K", 48)
            job_params["burn_in"] = job_params.get("burn_in", 1000)
            job_params["model_args"] = " MCMC/sample_a_start=100000 " + job_params.get("model_args", "")
            job_params["fast"] = True
            poll = 1

        # Issue a training command to the index.  Equivalent to:
        #
        # curl -X POST https://sturdystatistics.com/api/text/v1/index/{index_id}/train \
        #   -H "Content-Type: application/json" \
        #   -d '{
        #      "api_key": "API_KEY",
        #      PARAMS
        #    }'

        info = self._post(f"/{self.id}/train", job_params)
        job_id = info.json()["job_id"]
        job = Job(self.API_key, job_id, poll, _base_url=self._job_base_url())
        if wait:
            return job.wait()
        else:
            return job



    def predict(self, records: Iterable[Dict], batch_size: int = 1000) -> list[dict]:
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
        #   -d 10,
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
        max_excerpts_per_doc: int = 5,
        semantic_search_weight: float = .3,
        semantic_search_cutoff: float = .2,
        override_args: dict = dict(),
        return_df: bool = True
    ) -> pd.DataFrame:
        params = dict(
            offset=offset,
            limit=limit,
            sort_by=sort_by,
            ascending=ascending,
            filters=filters,
            context=context,
            max_excerpts_per_doc=max_excerpts_per_doc,
            semantic_search_weight=semantic_search_weight,
            semantic_search_cutoff=semantic_search_cutoff,
        )
        if search_query is not None:
            params["query"] = search_query
        if topic_id is not None:
            params['topic_ids'] = topic_id
        if topic_group_id is not None:
            params["topic_group_id"] = topic_group_id
        params = {**params, **override_args}
        res = self._get(f"/{self.id}/doc", params)
        res2 = res.json()["docs"]
        if not return_df: return res2
        elif len(res2) == 0: return pd.DataFrame(res2)
        res3 = [ r["metadata"] | r  for r in res2 ]
        df = pd.DataFrame(res3)
        front = [ "doc_id", "text" ]
        return df[[ *front, *[c for c in df.columns if c not in front ]]]
        

    def getDocs(
        self,
        doc_ids: list[str],
        search_query: Optional[str] = None,
        topic_id: Optional[int] = None,
        topic_group_id: Optional[int] = None,
        max_excerpts_per_doc: int = 5,
        context: int = 0,
        override_args: dict = dict(),
        return_df: bool = True,
    ) -> pd.DataFrame:
        assert len(doc_ids) > 0
        params = dict(context=context, max_excerpts_per_doc=max_excerpts_per_doc)
        if search_query is not None:
            params["query"] = search_query
        if topic_id is not None:
            params['topic_ids'] = topic_id
        if topic_group_id is not None:
            params["topic_group_id"] = topic_group_id
        params = {**params, **override_args}
        joined = ",".join(doc_ids)
        res = self._get(f"/{self.id}/doc/{joined}", params).json()
        res2 = res["docs"]
        if not return_df: return res2
        if len(res2) == 0: return pd.DataFrame(res2)
        res3 = [ r["metadata"] | r  for r in res2 ]
        df = pd.DataFrame(res3)
        front = [ "doc_id", "text" ]
        return df[[ *front, *[c for c in df.columns if c not in front ]]]

    def getDocsBinary(
        self,
        doc_ids: list[str],
    ):
        from spacy.tokens import Doc, Span, Token, DocBin
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
        field: str | None = None,
        override_args: dict = dict(),
    ) -> dict:
        params = dict(field=field)
        params = {**params, **override_args}
        if self.pandata is None:
            self.pandata = srsly.msgpack_loads(self._get(f"/{self.id}/pandata", params).content)
        return self.pandata # type: ignore

    def queryMeta(
            self,
            query: str, 
            search_query: str = "",
            semantic_search_weight: float = .3,
            semantic_search_cutoff: float = .2,
            override_args: dict = dict(),
            return_df: bool = True,
            paginate: bool = False,
    ) -> pd.DataFrame:
        params = dict(
            q=query,
            search_query=search_query,
            semantic_search_weight=semantic_search_weight,
            semantic_search_cutoff=semantic_search_cutoff,
        )
        params = {**params, **override_args}
        finalRes = []
        if not paginate:
            finalRes = srsly.msgpack_loads(self._get(f"/{self.id}/doc/meta", params).content)["results"] # type: ignore
        else:
            while True:
                params["q"] = query 
                params["offset"] = len(finalRes) 
                res: list[dict] = srsly.msgpack_loads(self._get(f"/{self.id}/doc/meta", params).content)["results"] # type: ignore
                if len(res) == 0: break
                finalRes.extend(res)

        if not return_df: return finalRes
        return pd.DataFrame(finalRes)
    
    def annotate(self):
        self._post(f"/{self.id}/annotate", dict())
        while True:
            res = self.get_status()
            if res["state"] == "ready":
                break
            time.sleep(3)

    def clone(self, new_name) -> dict:
        info = self._post(f"/{self.id}/clone", dict(new_name=new_name))
        job_id = info.json()["job_id"]
        job = Job(self.API_key, job_id, 5, _base_url=self._job_base_url())
        return job.wait()

    def delete(self, force: bool) -> dict:
        if not force:
            print("Are you sure you want to delete this index? There is no going back")
            return
        return self._post(f"/{self.id}/delete", dict()).json()

    def topicSearch(
        self,
        query: str = "",
        filters: str = "",
        limit: int = 100,
        semantic_search_weight: float = .3,
        semantic_search_cutoff: float = .2,
        override_args: dict = dict(),
        return_df: bool = True
    ) -> pd.DataFrame:
        params = dict(
            query=query,
            filters=filters,
            limit=limit,
            semantic_search_weight=semantic_search_weight,
            semantic_search_cutoff=semantic_search_cutoff,
        )
        params = {**params, **override_args}
        res = self._get(f"/{self.id}/topic/search", params).json()["topics"]
        if not return_df: return res
        return pd.DataFrame(res)


    def topicDiff(
        self,
        filter1: str = "",
        filter2: str = "",
        search_query1: str = "",
        search_query2: str = "",
        limit: int = 50,
        cutoff: float = 1.0,
        min_confidence: float = 95,
        semantic_search_weight: float = .3,
        semantic_search_cutoff: float = .2,
        override_args: dict = dict(),
        return_df: bool = True
    ) -> pd.DataFrame:
        params = dict(
            filter1=filter1,
            filter2=filter2,
            limit=limit,
            cutoff=cutoff,
            min_confidence=min_confidence,
            search_query1=search_query1,
            search_query2=search_query2,
            semantic_search_weight=semantic_search_weight,
            semantic_search_cutoff=semantic_search_cutoff,
        )
        params = {**params, **override_args}
        res = self._get(f"/{self.id}/topic/diff", params).json()["topics"]
        if not return_df: return res
        return pd.DataFrame(res)

    def topicWords(
        self,
        override_args: dict = dict(),
        return_df: bool = True
    ) -> pd.DataFrame:
        params = dict(
        )
        params = {**params, **override_args}
        res = self._get(f"/{self.id}/topic/words", params).json()["topics"]
        if not return_df: return res
        return pd.DataFrame(res)

    def docTopics(
        self,
        doc: str,
        override_args: dict = dict(),
        return_df: bool = True
    ) -> pd.DataFrame:
        params = dict(
        )
        params = dict(doc=doc)
        params = {**params, **override_args}
        res = self._get(f"/{self.id}/topic/doc", params).json()["topics"]
        if not return_df: return res
        return pd.DataFrame(res)

    def listJobs(
        self,
        status: str= "RUNNING",
        job_name: Optional[str] = None,
        only_current_index: bool = True,
        return_df: bool = True,
    ) -> pd.DataFrame:
        assert status in [None, "", "RUNNING", "FAILED", "SUCCEEDED", "PENDING", "CANCELLED"]
        params = dict()
        if only_current_index:
            params["index_id"] = self.id
        if status is not None and status.strip() != "":
            params["status"] = status
        if job_name is not None and job_name.strip() != "":
            params["job_name"] = job_name

        job = Job(self.API_key, "", 1, _base_url=self._job_base_url())
        res = job._get("", params).json()
        if not return_df: return res
        return pd.DataFrame(res)

    def listIndices(
        self,
        name_filter: Optional[str] = None,
        state_filter: Optional[str] = None,
        return_df: bool = True,
    ) -> pd.DataFrame:
        res = self._get("", dict()).json()
        results = []
        for r in res:
            if name_filter is not None and name_filter not in r["name"]:
                continue
            if state_filter is not None and state_filter != r["state"]:
                continue
            results.append(r)
        if not return_df: return results
        return pd.DataFrame(results)

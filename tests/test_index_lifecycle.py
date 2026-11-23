from unittest.mock import Mock

import pytest

from sturdystats import Index, SturdyStatsSdkError


@pytest.fixture
def index():
    return Index(org_id="o-example", api_key="test-key",
                 base_url="https://example.test", id="ix-example")


@pytest.mark.parametrize(("method", "status"), [("archive", "archived"), ("restore", "ready")])
def test_lifecycle_actions_return_index_metadata(index, monkeypatch, method, status):
    metadata = {"id": index.id, "status": status}
    if status == "archived":
        metadata["archived-at"] = "2026-11-22T10:00:00Z"
    post = Mock(return_value=Mock(ok=True, json=Mock(return_value=metadata)))
    monkeypatch.setattr(index._session, "post", post)

    assert getattr(index, method)() == metadata
    post.assert_called_once_with(
        f"https://example.test/api/v1/orgs/o-example/indices/ix-example/{method}", json={}
    )


@pytest.mark.parametrize(("method", "code"), [("archive", 429), ("restore", 409)])
def test_lifecycle_actions_preserve_api_errors(index, monkeypatch, method, code):
    post = Mock(return_value=Mock(ok=False, status_code=code, text="lifecycle conflict"))
    monkeypatch.setattr(index._session, "post", post)

    with pytest.raises(SturdyStatsSdkError, match="lifecycle conflict") as error:
        getattr(index, method)()
    assert error.value.status_code == code


def test_list_accepts_status_and_preserves_the_positional_transform(index, monkeypatch):
    records = [{"id": index.id, "status": "archived", "archived-at": "2026-11-22T10:00:00Z"}]
    get = Mock(return_value=Mock(ok=True, json=Mock(return_value={"indices": records})))
    monkeypatch.setattr(index._session, "get", get)
    url = "https://example.test/api/v1/orgs/o-example/indices"

    assert index.list(status="archived").to_dict("records") == records
    get.assert_called_once_with(url, params={"status": "archived"})
    get.reset_mock()

    assert index.list(lambda frame: frame["id"].tolist()) == [index.id]
    get.assert_called_once_with(url, params=None)


def test_wait_stops_when_an_index_is_archived(index, monkeypatch):
    get = Mock(return_value={"status": "archived"})
    monkeypatch.setattr(index, "_get", get)
    sleep = Mock(side_effect=AssertionError("archived status must end polling"))
    monkeypatch.setattr("time.sleep", sleep)

    with pytest.raises(SturdyStatsSdkError, match="is archived. Restore it"):
        index.wait()
    get.assert_called_once_with("indices/ix-example")
    sleep.assert_not_called()

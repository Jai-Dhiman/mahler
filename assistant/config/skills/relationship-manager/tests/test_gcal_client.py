import json
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import gcal_client


def _make_gcal_response(items, next_page_token=None):
    body = {"items": items}
    if next_page_token:
        body["nextPageToken"] = next_page_token
    raw = json.dumps(body).encode()
    resp = MagicMock()
    resp.read.return_value = raw
    resp.__enter__ = lambda s: s
    resp.__exit__ = MagicMock(return_value=False)
    return resp


def _event_fixture(event_id="evt1", summary="Team Sync", attendee_emails=None, self_attendee=None):
    attendees = []
    for email in (attendee_emails or []):
        attendees.append({"email": email})
    if self_attendee:
        attendees.append({"email": self_attendee, "self": True})
    item = {
        "id": event_id,
        "summary": summary,
        "start": {"dateTime": "2026-04-20T10:00:00Z"},
        "end": {"dateTime": "2026-04-20T11:00:00Z"},
    }
    if attendees:
        item["attendees"] = attendees
    return item


def test_list_events_paginates():
    evt1 = _event_fixture(event_id="evt1", summary="Meeting A")
    evt2 = _event_fixture(event_id="evt2", summary="Meeting B")
    with patch("gcal_client._OPENER") as mock_opener:
        mock_opener.open.side_effect = [
            _make_gcal_response(items=[evt1], next_page_token="token_page2"),
            _make_gcal_response(items=[evt2]),
        ]
        results = gcal_client.list_events(
            access_token="tok",
            time_min="2026-04-01T00:00:00Z",
            time_max="2026-04-30T23:59:59Z",
        )
    assert len(results) == 2
    assert results[0]["id"] == "evt1"
    assert results[0]["summary"] == "Meeting A"
    assert results[1]["id"] == "evt2"
    assert results[1]["summary"] == "Meeting B"
    assert mock_opener.open.call_count == 2
    # Second call must include pageToken in the URL
    second_url = mock_opener.open.call_args_list[1][0][0].full_url
    assert "pageToken=token_page2" in second_url


def test_list_events_single_page():
    evt = _event_fixture(event_id="evt1", summary="Solo Event")
    with patch("gcal_client._OPENER") as mock_opener:
        mock_opener.open.return_value = _make_gcal_response(items=[evt])
        results = gcal_client.list_events(
            access_token="tok",
            time_min="2026-04-01T00:00:00Z",
            time_max="2026-04-30T23:59:59Z",
        )
    assert len(results) == 1
    assert mock_opener.open.call_count == 1


def test_list_events_empty():
    with patch("gcal_client._OPENER") as mock_opener:
        mock_opener.open.return_value = _make_gcal_response(items=[])
        results = gcal_client.list_events(
            access_token="tok",
            time_min="2026-04-01T00:00:00Z",
            time_max="2026-04-30T23:59:59Z",
        )
    assert results == []


def test_normalize_event_filters_self_attendee():
    evt = _event_fixture(
        event_id="evt1",
        summary="Planning",
        attendee_emails=["alice@example.com", "bob@example.com"],
        self_attendee="me@example.com",
    )
    with patch("gcal_client._OPENER") as mock_opener:
        mock_opener.open.return_value = _make_gcal_response(items=[evt])
        results = gcal_client.list_events(
            access_token="tok",
            time_min="2026-04-01T00:00:00Z",
            time_max="2026-04-30T23:59:59Z",
        )
    assert len(results) == 1
    attendees = results[0]["attendees"]
    assert "me@example.com" not in attendees
    assert "alice@example.com" in attendees
    assert "bob@example.com" in attendees


def test_normalize_event_no_attendees():
    evt = _event_fixture(event_id="evt1", summary="Focus Block")
    with patch("gcal_client._OPENER") as mock_opener:
        mock_opener.open.return_value = _make_gcal_response(items=[evt])
        results = gcal_client.list_events(
            access_token="tok",
            time_min="2026-04-01T00:00:00Z",
            time_max="2026-04-30T23:59:59Z",
        )
    assert results[0]["attendees"] == []

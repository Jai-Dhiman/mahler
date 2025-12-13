from __future__ import annotations

"""HTTP client wrapper for Cloudflare Workers Python.

Uses the JavaScript fetch API via Pyodide interop.
"""

import json

from js import Headers, Object, fetch
from pyodide.ffi import to_js


async def request(
    method: str,
    url: str,
    headers: dict | None = None,
    params: dict | None = None,
    json_data: dict | None = None,
    timeout: float = 30.0,
) -> dict:
    """Make an HTTP request using the Workers fetch API.

    Args:
        method: HTTP method (GET, POST, PUT, DELETE, etc.)
        url: Full URL to request
        headers: Optional headers dict
        params: Optional query parameters
        json_data: Optional JSON body
        timeout: Request timeout in seconds (not fully supported in Workers)

    Returns:
        Parsed JSON response as dict

    Raises:
        Exception: On HTTP errors or network failures
    """
    # Build URL with query params
    if params:
        query_string = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{url}?{query_string}"

    # Build headers
    js_headers = Headers.new()
    if headers:
        for key, value in headers.items():
            js_headers.append(key, value)

    # Build fetch options
    options = {
        "method": method,
        "headers": js_headers,
    }

    if json_data is not None:
        options["body"] = json.dumps(json_data)

    # Convert to JS object
    js_options = to_js(options, dict_converter=Object.fromEntries)

    # Make request
    response = await fetch(url, js_options)

    # Check for errors
    if not response.ok:
        text = await response.text()
        raise Exception(f"HTTP {response.status}: {text}")

    # Handle empty responses
    if response.status == 204:
        return {}

    # Parse JSON response
    text = await response.text()
    if not text:
        return {}

    return json.loads(text)


async def get(url: str, headers: dict | None = None, params: dict | None = None) -> dict:
    """Make a GET request."""
    return await request("GET", url, headers=headers, params=params)


async def post(url: str, headers: dict | None = None, json_data: dict | None = None) -> dict:
    """Make a POST request."""
    return await request("POST", url, headers=headers, json_data=json_data)


async def put(url: str, headers: dict | None = None, json_data: dict | None = None) -> dict:
    """Make a PUT request."""
    return await request("PUT", url, headers=headers, json_data=json_data)


async def delete(url: str, headers: dict | None = None) -> dict:
    """Make a DELETE request."""
    return await request("DELETE", url, headers=headers)


async def ping_heartbeat(
    base_url: str | None,
    job_name: str,
    success: bool = True,
    message: str | None = None,
) -> bool:
    """Ping external heartbeat monitoring service.

    Supports Healthchecks.io and Cronitor style endpoints.

    Healthchecks.io format: https://hc-ping.com/<uuid>/<job_name>
    - Success: GET to base URL
    - Failure: GET to base URL + /fail
    - Start: GET to base URL + /start

    Cronitor format: https://cronitor.link/p/<api_key>/<job_name>
    - Success: ?state=complete
    - Failure: ?state=fail

    Args:
        base_url: Base heartbeat URL (e.g., https://hc-ping.com/<uuid>)
        job_name: Name of the job being monitored
        success: Whether the job succeeded
        message: Optional status message

    Returns:
        True if ping was sent successfully, False otherwise
    """
    if not base_url:
        return True  # No heartbeat configured, skip

    try:
        # Build URL based on service type and status
        if "hc-ping.com" in base_url or "healthchecks.io" in base_url:
            # Healthchecks.io format
            url = f"{base_url}/{job_name}"
            if not success:
                url = f"{url}/fail"
        elif "cronitor.link" in base_url or "cronitor.io" in base_url:
            # Cronitor format
            state = "complete" if success else "fail"
            url = f"{base_url}/{job_name}?state={state}"
            if message:
                url = f"{url}&message={message}"
        else:
            # Generic format - append job name and status
            url = f"{base_url}/{job_name}"
            if not success:
                url = f"{url}?status=fail"

        # Make the ping request
        js_headers = Headers.new()
        js_options = to_js({"method": "GET", "headers": js_headers}, dict_converter=Object.fromEntries)
        response = await fetch(url, js_options)

        return response.ok

    except Exception as e:
        print(f"Heartbeat ping failed for {job_name}: {e}")
        return False


async def ping_heartbeat_start(base_url: str | None, job_name: str) -> bool:
    """Signal start of a monitored job.

    For services that support start/end tracking (like Healthchecks.io),
    this marks the beginning of a job execution.
    """
    if not base_url:
        return True

    try:
        if "hc-ping.com" in base_url or "healthchecks.io" in base_url:
            url = f"{base_url}/{job_name}/start"
            js_headers = Headers.new()
            js_options = to_js({"method": "GET", "headers": js_headers}, dict_converter=Object.fromEntries)
            response = await fetch(url, js_options)
            return response.ok
        return True  # Other services don't support start signal
    except Exception as e:
        print(f"Heartbeat start ping failed for {job_name}: {e}")
        return False

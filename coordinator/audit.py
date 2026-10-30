""" 
Audit logging for the MIMIC-DP coordinator 

The audit log records  metadata about any query execution without 
storing raw patient data, hospital level statistics, or exact pre-DP aggregate values

Records are stored as json lines (JSONL) with one JSON object per audit event

"""

import json
import os 
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import uuid4

#-----------------------------------------
###############Audit confiduration########
#-----------------------------------------

AUDIT_LOG_PATH = Path (
    os.getenv(
             "AUDIT_LOG_PATH",
             "results/audit/query_audit.jsonl",
              )
)

_AUDIT_LOCK = Lock()

#-----------------------------------------
###############Audit writer########
#-----------------------------------------

def log_query_event(
    *,
   query_id: str,
    status: str,
    mode: str,
    request_id: str | None = None,
    epsilon: float | None = None,
    delta: float | None = None,
    participating_hospitals: int | None = None,
    error: str | None = None,
) -> dict:
    
    """
    Write one query-execution event to the audit log.

    Parameters
    ----------
    query_id:
        Approved query identifier, e.g. ``Q001``.

    status:
        Execution outcome, typically ``success`` or ``failed``.

    mode:
        Execution mode, e.g. ``plain_baseline`` or ``central_dp``.

    request_id:
        Identifier linking audit events to a coordinator request.

    epsilon, delta:
        Privacy parameters. These remain ``None`` for the current
        non-private baseline.

    participating_hospitals:
        Number of hospital nodes involved in the query.

    error:
        Non-sensitive error description when execution fails.

    Returns
    -------
    dict
        The audit record written to disk.
    """

    AUDIT_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if request_id is None:
        request_id = str(uuid4())

    record = {
        "event_id": str(uuid4()),
        "request_id": request_id,
        "timestamp_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "query_id": query_id.upper(),
        "mode": mode,
        "status": status,
        "participating_hospitals": participating_hospitals,
        "epsilon": epsilon,
        "delta": delta,
        "error": error,
    }

    with _AUDIT_LOCK:
        with AUDIT_LOG_PATH.open(
            "a",
            encoding="utf-8",
        ) as audit_file:
            audit_file.write(
                json.dumps(record) + "\n"
            )

    return record
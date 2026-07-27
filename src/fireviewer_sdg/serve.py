"""Authenticated production API and in-pod visual review console."""

from __future__ import annotations

import hmac
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from fireviewer_sdg.preparation_progress import load_progress
from fireviewer_sdg.production import ProductionManager
from fireviewer_sdg.review_store import CaseStore
from fireviewer_sdg.training_release import TrainingReleaseLocked, build_training_release


STATIC_ROOT = Path(__file__).resolve().parent / "static"
MAXIMUM_REQUEST_BYTES = 8192


def create_server(
    *,
    host: str,
    port: int,
    auth_token: str,
    volume_root: Path,
    status: dict[str, object],
    campaign_path: Path,
    production_manager: ProductionManager,
    case_store: CaseStore,
) -> ThreadingHTTPServer:
    if len(auth_token) < 32:
        raise RuntimeError("FW_SDG_AUTH_TOKEN must contain at least 32 characters")

    def worker_snapshot() -> dict[str, Any]:
        payload: dict[str, Any] = dict(status)
        preparation = payload.get("input_preparation")
        preparation_payload = (
            dict(preparation) if isinstance(preparation, dict) else {}
        )
        progress = load_progress(volume_root)
        if progress is not None:
            preparation_payload["progress"] = progress
        if preparation_payload:
            payload["input_preparation"] = preparation_payload
        return payload

    class Handler(BaseHTTPRequestHandler):
        server_version = "FireViewerSDG/0.4"

        def _authenticated(self) -> bool:
            supplied = self.headers.get("Authorization", "")
            expected = f"Bearer {auth_token}"
            return hmac.compare_digest(supplied, expected)

        def _security_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; img-src 'self' blob:; script-src 'self'; "
                "style-src 'self'; style-src-attr 'unsafe-inline'; "
                "connect-src 'self'; base-uri 'none'; "
                "form-action 'self'; frame-ancestors 'none'",
            )

        def _write_bytes(
            self, code: HTTPStatus, body: bytes, *, content_type: str
        ) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self._security_headers()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _write(self, code: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            self._write_bytes(code, body, content_type="application/json; charset=utf-8")

        def _require_authentication(self) -> bool:
            if self._authenticated():
                return True
            self._write(HTTPStatus.UNAUTHORIZED, {"error": "unauthorized"})
            return False

        def _read_json(self) -> dict[str, Any]:
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ValueError("invalid_content_length") from exc
            if not 1 <= content_length <= MAXIMUM_REQUEST_BYTES:
                raise ValueError("invalid_request_size")
            if self.headers.get_content_type() != "application/json":
                raise ValueError("content_type_must_be_application_json")
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict):
                raise ValueError("request_body_must_be_an_object")
            return payload

        def _production_inputs_ready(self, *, stage: str) -> tuple[bool, str]:
            preparation = status.get("input_preparation")
            if not isinstance(preparation, dict):
                return False, "input preparation has not produced a reviewed pilot catalog"
            state = preparation.get("state")
            if state not in {"existing", "prepared"}:
                reason = str(
                    preparation.get("reason")
                    or f"input preparation is {state or 'incomplete'}"
                )
                return False, reason
            if stage == "bulk" and preparation.get("bulk_allowed") is not True:
                return (
                    False,
                    "the three-site setup is pilot-only; add the planned geographic "
                    "sites and regenerate the locked catalog before bulk production",
                )
            return True, ""

        def _serve_static(self, name: str) -> None:
            allowed = {"console.html", "console.css", "console.js"}
            if name not in allowed:
                self._write(HTTPStatus.NOT_FOUND, {"error": "not_found"})
                return
            path = STATIC_ROOT / name
            if not path.is_file():
                self._write(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "console_asset_absent"})
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            if name == "console.html":
                content_type = "text/html; charset=utf-8"
            elif name.endswith(".js"):
                content_type = "text/javascript; charset=utf-8"
            elif name.endswith(".css"):
                content_type = "text/css; charset=utf-8"
            self._write_bytes(HTTPStatus.OK, path.read_bytes(), content_type=content_type)

        def _case_route(self, path: str) -> tuple[str, str, str] | None:
            parts = path.strip("/").split("/")
            if len(parts) != 5 or parts[:2] != ["v1", "cases"]:
                return None
            return parts[2], parts[3], parts[4]

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urlparse(self.path)
            path = parsed.path
            if path == "/":
                self._write(
                    HTTPStatus.OK,
                    {
                        "service": "fireviewer-sdg",
                        "status": "alive",
                        "console": "/console",
                    },
                )
                return
            if path == "/healthz":
                self._write(HTTPStatus.OK, {"status": "alive"})
                return
            if path == "/favicon.ico":
                self._write_bytes(
                    HTTPStatus.NO_CONTENT,
                    b"",
                    content_type="image/x-icon",
                )
                return
            if path in {"/console", "/console/"}:
                self._serve_static("console.html")
                return
            if path == "/static/console.css":
                self._serve_static("console.css")
                return
            if path == "/static/console.js":
                self._serve_static("console.js")
                return
            if not self._require_authentication():
                return
            if path == "/readyz":
                self._write(HTTPStatus.OK, worker_snapshot())
                return
            if path == "/v1/status":
                receipt = volume_root / "provision" / "receipts" / "latest.json"
                payload: dict[str, Any] = worker_snapshot()
                payload["production"] = production_manager.snapshot()
                payload["deliverables"] = case_store.status()
                if receipt.is_file():
                    payload["provision_receipt"] = json.loads(
                        receipt.read_text(encoding="utf-8")
                    )
                self._write(HTTPStatus.OK, payload)
                return
            if path == "/v1/console/status":
                self._write(
                    HTTPStatus.OK,
                    {
                        "worker": worker_snapshot(),
                        "production": production_manager.snapshot(),
                        "deliverables": case_store.status(),
                    },
                )
                return
            if path in {"/v1/production/status", "/v1/campaign/status"}:
                self._write(HTTPStatus.OK, production_manager.snapshot())
                return
            if path == "/v1/production/preview":
                production = production_manager.snapshot()
                current = production.get("current_batch")
                progress = (
                    current.get("progress")
                    if isinstance(current, dict)
                    else None
                )
                last_completed = (
                    progress.get("last_completed")
                    if isinstance(progress, dict)
                    else None
                )
                relative = (
                    str(last_completed.get("preview_relpath", "")).strip()
                    if isinstance(last_completed, dict)
                    else ""
                )
                preview = (volume_root / relative).resolve()
                if (
                    not relative
                    or volume_root not in preview.parents
                    or not preview.is_file()
                ):
                    self._write(
                        HTTPStatus.NOT_FOUND,
                        {"error": "live_preview_not_available"},
                    )
                    return
                content_type = (
                    mimetypes.guess_type(preview.name)[0]
                    or "application/octet-stream"
                )
                if not content_type.startswith("image/"):
                    self._write(
                        HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                        {"error": "live_preview_is_not_an_image"},
                    )
                    return
                self._write_bytes(
                    HTTPStatus.OK,
                    preview.read_bytes(),
                    content_type=content_type,
                )
                return
            if path == "/v1/cases":
                query = parse_qs(parsed.query, keep_blank_values=False)
                try:
                    payload = case_store.list(
                        category=query.get("category", [""])[0],
                        offset=int(query.get("offset", ["0"])[0]),
                        limit=int(query.get("limit", ["50"])[0]),
                        review_state=query.get("review_state", [None])[0],
                    )
                except (TypeError, ValueError) as exc:
                    self._write(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                self._write(HTTPStatus.OK, payload)
                return
            if path == "/v1/logs":
                query = parse_qs(parsed.query, keep_blank_values=False)
                try:
                    tail = int(query.get("tail", ["200"])[0])
                    events = case_store.tail_events(tail)
                except (TypeError, ValueError) as exc:
                    self._write(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                self._write(HTTPStatus.OK, {"events": events})
                return
            case_route = self._case_route(path)
            if case_route and case_route[2] == "preview":
                category, case_id, _ = case_route
                try:
                    preview = case_store.preview(category, case_id)
                except (KeyError, ValueError):
                    self._write(HTTPStatus.NOT_FOUND, {"error": "case_not_found"})
                    return
                content_type = mimetypes.guess_type(preview.name)[0] or "application/octet-stream"
                self._write_bytes(HTTPStatus.OK, preview.read_bytes(), content_type=content_type)
                return
            self._write(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            parsed = urlparse(self.path)
            path = parsed.path
            if not self._require_authentication():
                return
            if path in {"/v1/production/pilot", "/v1/campaign/start"}:
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self._write(HTTPStatus.BAD_REQUEST, {"error": "invalid_content_length"})
                    return
                if content_length != 0:
                    self._write(HTTPStatus.BAD_REQUEST, {"error": "request_body_not_allowed"})
                    return
                inputs_ready, reason = self._production_inputs_ready(stage="pilot")
                if not inputs_ready:
                    self._write(
                        HTTPStatus.CONFLICT,
                        {"error": f"pilot_locked: {reason}"},
                    )
                    return
                try:
                    production_manager.start_pilot(campaign_path)
                except RuntimeError as exc:
                    self._write(HTTPStatus.CONFLICT, {"error": str(exc)})
                    return
                self._write(HTTPStatus.ACCEPTED, production_manager.snapshot())
                return
            if path == "/v1/production/bulk":
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self._write(HTTPStatus.BAD_REQUEST, {"error": "invalid_content_length"})
                    return
                if content_length != 0:
                    self._write(HTTPStatus.BAD_REQUEST, {"error": "request_body_not_allowed"})
                    return
                inputs_ready, reason = self._production_inputs_ready(stage="bulk")
                if not inputs_ready:
                    self._write(
                        HTTPStatus.CONFLICT,
                        {"error": f"bulk_locked: {reason}"},
                    )
                    return
                try:
                    production_manager.continue_bulk(campaign_path)
                except RuntimeError as exc:
                    self._write(HTTPStatus.CONFLICT, {"error": str(exc)})
                    return
                self._write(HTTPStatus.ACCEPTED, production_manager.snapshot())
                return
            if path == "/v1/training/release":
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self._write(HTTPStatus.BAD_REQUEST, {"error": "invalid_content_length"})
                    return
                if content_length != 0:
                    self._write(HTTPStatus.BAD_REQUEST, {"error": "request_body_not_allowed"})
                    return
                try:
                    release = build_training_release(case_store)
                except TrainingReleaseLocked as exc:
                    self._write(HTTPStatus.CONFLICT, {"error": str(exc)})
                    return
                self._write(HTTPStatus.CREATED, release)
                return
            case_route = self._case_route(path)
            if case_route and case_route[2] == "review":
                category, case_id, _ = case_route
                try:
                    body = self._read_json()
                    review = case_store.review(
                        category=category,
                        case_id=case_id,
                        decision=str(body.get("decision", "")),
                        reviewer=str(body.get("reviewer", "")),
                        notes=str(body.get("notes", "")),
                        quality_checks=body.get("quality_checks"),
                    )
                except KeyError:
                    self._write(HTTPStatus.NOT_FOUND, {"error": "case_not_found"})
                    return
                except (OSError, ValueError, json.JSONDecodeError) as exc:
                    self._write(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
                    return
                self._write(HTTPStatus.OK, review)
                return
            self._write(HTTPStatus.NOT_FOUND, {"error": "not_found"})

        def log_message(self, format: str, *args: object) -> None:
            print(f"sdg-http {self.address_string()} {format % args}", flush=True)

    return ThreadingHTTPServer((host, port), Handler)


def serve(
    *,
    port: int,
    auth_token: str,
    volume_root: Path,
    status: dict[str, object],
    campaign_path: Path,
    production_manager: ProductionManager,
    case_store: CaseStore,
) -> None:
    create_server(
        host="0.0.0.0",
        port=port,
        auth_token=auth_token,
        volume_root=volume_root,
        status=status,
        campaign_path=campaign_path,
        production_manager=production_manager,
        case_store=case_store,
    ).serve_forever()

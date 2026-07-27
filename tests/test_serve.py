from __future__ import annotations

import http.client
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fireviewer_sdg.serve import create_server  # noqa: E402
from fireviewer_sdg.preparation_progress import write_progress  # noqa: E402
from fireviewer_sdg.review_store import CaseStore  # noqa: E402


READY_STATUS = {
    "status": "ready",
    "input_preparation": {
        "state": "prepared",
        "site_count": 3,
        "production_scope": "pilot_setup_proof",
        "bulk_allowed": False,
    },
}


class _Manager:
    def __init__(self) -> None:
        self.started: Path | None = None
        self.bulk: Path | None = None

    def start_pilot(self, path: Path) -> None:
        self.started = path

    def continue_bulk(self, path: Path) -> None:
        self.bulk = path

    def snapshot(self) -> dict[str, object]:
        return {"state": "queued" if self.started else "idle"}


class ServiceTests(unittest.TestCase):
    def test_console_status_reads_persistent_input_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_progress(
                root,
                phase="ign_terrain_download",
                message="Préparation IGN du site 2/3.",
                sites_completed=1,
                sites_total=3,
            )
            server = create_server(
                host="127.0.0.1",
                port=0,
                auth_token="p" * 32,
                volume_root=root,
                status={
                    "status": "ready",
                    "input_preparation": {
                        "state": "preparing",
                        "phase": "ign_terrain_and_event_catalog",
                    },
                },
                campaign_path=root / "campaign.json",
                production_manager=_Manager(),  # type: ignore[arg-type]
                case_store=CaseStore(root),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_address[1], timeout=5
            )
            try:
                connection.request(
                    "GET",
                    "/v1/console/status",
                    headers={"Authorization": f"Bearer {'p' * 32}"},
                )
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                progress = json.loads(response.read())["worker"][
                    "input_preparation"
                ]["progress"]
                self.assertEqual(progress["state"], "running")
                self.assertEqual(progress["phase"], "ign_terrain_download")
                self.assertEqual(progress["sites_completed"], 1)
                self.assertEqual(progress["sites_total"], 3)
            finally:
                connection.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_health_authentication_and_campaign_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            campaign_path = root / "campaign.json"
            campaign_path.write_text("{}", encoding="ascii")
            manager = _Manager()
            server = create_server(
                host="127.0.0.1",
                port=0,
                auth_token="a" * 32,
                volume_root=root,
                status=READY_STATUS,
                campaign_path=campaign_path,
                production_manager=manager,  # type: ignore[arg-type]
                case_store=CaseStore(root),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_address[1], timeout=5
            )
            try:
                connection.request("GET", "/healthz")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                response.read()

                connection.request("GET", "/v1/campaign/status")
                response = connection.getresponse()
                self.assertEqual(response.status, 401)
                response.read()

                headers = {"Authorization": f"Bearer {'a' * 32}"}
                connection.request("POST", "/v1/campaign/start", headers=headers)
                response = connection.getresponse()
                self.assertEqual(response.status, 202)
                payload = json.loads(response.read())
                self.assertEqual(payload["state"], "queued")
                self.assertEqual(manager.started, campaign_path)
            finally:
                connection.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_start_endpoint_rejects_request_body(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = _Manager()
            server = create_server(
                host="127.0.0.1",
                port=0,
                auth_token="b" * 32,
                volume_root=root,
                status=READY_STATUS,
                campaign_path=root / "campaign.json",
                production_manager=manager,  # type: ignore[arg-type]
                case_store=CaseStore(root),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_address[1], timeout=5
            )
            try:
                connection.request(
                    "POST",
                    "/v1/campaign/start",
                    body=b"{}",
                    headers={"Authorization": f"Bearer {'b' * 32}"},
                )
                response = connection.getresponse()
                self.assertEqual(response.status, 400)
                response.read()
                self.assertIsNone(manager.started)
            finally:
                connection.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_three_site_proof_blocks_bulk_at_the_api(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = _Manager()
            server = create_server(
                host="127.0.0.1",
                port=0,
                auth_token="d" * 32,
                volume_root=root,
                status=READY_STATUS,
                campaign_path=root / "campaign.json",
                production_manager=manager,  # type: ignore[arg-type]
                case_store=CaseStore(root),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_address[1], timeout=5
            )
            try:
                connection.request(
                    "POST",
                    "/v1/production/bulk",
                    headers={"Authorization": f"Bearer {'d' * 32}"},
                )
                response = connection.getresponse()
                self.assertEqual(response.status, 409)
                payload = json.loads(response.read())
                self.assertIn("three-site setup is pilot-only", payload["error"])
                self.assertIsNone(manager.bulk)
            finally:
                connection.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_console_shell_is_public_but_case_inventory_is_authenticated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server = create_server(
                host="127.0.0.1",
                port=0,
                auth_token="c" * 32,
                volume_root=root,
                status=READY_STATUS,
                campaign_path=root / "campaign.json",
                production_manager=_Manager(),  # type: ignore[arg-type]
                case_store=CaseStore(root),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_address[1], timeout=5
            )
            try:
                connection.request("GET", "/console")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertIn(b"Console de production", response.read())

                connection.request("GET", "/v1/cases?category=terrestrial_fire_points")
                response = connection.getresponse()
                self.assertEqual(response.status, 401)
                response.read()

                connection.request(
                    "GET",
                    "/v1/console/status",
                    headers={"Authorization": f"Bearer {'c' * 32}"},
                )
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                payload = json.loads(response.read())
                self.assertEqual(payload["deliverables"]["target_total"], 16384)
                self.assertFalse(payload["deliverables"]["export_ready"])
                self.assertEqual(payload["production"]["state"], "idle")

                connection.request(
                    "GET",
                    "/v1/cases?category=terrestrial_fire_points&offset=0&limit=50",
                    headers={"Authorization": f"Bearer {'c' * 32}"},
                )
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                inventory = json.loads(response.read())
                self.assertEqual(inventory["total"], 0)
                self.assertEqual(inventory["items"], [])

                connection.request(
                    "GET",
                    "/v1/logs?tail=50",
                    headers={"Authorization": f"Bearer {'c' * 32}"},
                )
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(json.loads(response.read())["events"], [])
            finally:
                connection.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

    def test_live_production_preview_uses_only_current_progress_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            preview = root / "production" / "generated" / "latest.png"
            preview.parent.mkdir(parents=True)
            preview.write_bytes(b"\x89PNG\r\n\x1a\nactual-preview")

            class LiveManager(_Manager):
                def snapshot(self) -> dict[str, object]:
                    return {
                        "state": "running",
                        "current_batch": {
                            "progress": {
                                "last_completed": {
                                    "case_id": "tfp-000000",
                                    "preview_relpath": (
                                        preview.relative_to(root).as_posix()
                                    ),
                                }
                            }
                        },
                    }

            server = create_server(
                host="127.0.0.1",
                port=0,
                auth_token="e" * 32,
                volume_root=root,
                status=READY_STATUS,
                campaign_path=root / "campaign.json",
                production_manager=LiveManager(),  # type: ignore[arg-type]
                case_store=CaseStore(root),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            connection = http.client.HTTPConnection(
                "127.0.0.1", server.server_address[1], timeout=5
            )
            try:
                connection.request(
                    "GET",
                    "/v1/production/preview",
                    headers={"Authorization": f"Bearer {'e' * 32}"},
                )
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(
                    response.getheader("Content-Type"),
                    "image/png",
                )
                self.assertEqual(
                    response.read(),
                    b"\x89PNG\r\n\x1a\nactual-preview",
                )
            finally:
                connection.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()

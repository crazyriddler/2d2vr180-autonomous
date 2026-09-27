import hashlib
import http.server
import threading

import pytest

from twod2vr180.models import LicenseNotAccepted, ModelEntry, ModelError, ModelFile, ModelManager

PAYLOAD = bytes(range(256)) * 4096  # 1 MiB


class RangeHandler(http.server.BaseHTTPRequestHandler):
    fail_after = None  # bytes after which the connection is dropped (once)

    def log_message(self, *a):
        pass

    def do_GET(self):
        data = PAYLOAD
        start = 0
        rng = self.headers.get("Range")
        if rng:
            start = int(rng.split("=")[1].split("-")[0])
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        else:
            self.send_response(200)
        body = data[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if RangeHandler.fail_after is not None:
            cut = RangeHandler.fail_after
            RangeHandler.fail_after = None
            self.wfile.write(body[:cut])
            self.wfile.flush()
            self.connection.close()
            return
        self.wfile.write(body)


@pytest.fixture
def server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/model.bin"
    srv.shutdown()


def entry(url, sha=None, size=None):
    return ModelEntry(id="m", display_name="M", source="test", revision="r1", license="Test-1.0",
                      license_url="http://x", commercial_use=True, redistributable=False, required_vram_gb=1,
                      backends=[], files=[ModelFile("model.bin", url, sha, size)])


def test_license_gate(tmp_path, server):
    mm = ModelManager(tmp_path, [entry(server)])
    with pytest.raises(LicenseNotAccepted):
        mm.download("m")


def test_download_verify_and_tofu(tmp_path, server):
    mm = ModelManager(tmp_path, [entry(server)])
    mm.accept_license("m")
    mm.download("m")
    assert mm.is_installed("m")
    assert mm.status("m")["hash_status"] == "tofu"
    assert mm.verify("m") == []
    # corruption is detected
    p = mm.paths("m")["model.bin"]
    raw = bytearray(p.read_bytes())
    raw[10] ^= 0xFF
    p.write_bytes(bytes(raw))
    assert any("SHA256" in x for x in mm.verify("m"))


def test_pinned_hash_mismatch_discards(tmp_path, server):
    mm = ModelManager(tmp_path, [entry(server, sha="0" * 64)])
    mm.accept_license("m")
    with pytest.raises(ModelError, match="SHA256 mismatch"):
        mm.download("m")
    assert not mm.is_installed("m")
    assert not (tmp_path / "m" / "model.bin").exists()


def test_resume_after_interruption(tmp_path, server):
    sha = hashlib.sha256(PAYLOAD).hexdigest()
    mm = ModelManager(tmp_path, [entry(server, sha=sha, size=len(PAYLOAD))])
    mm.accept_license("m")
    RangeHandler.fail_after = 300_000
    with pytest.raises(ModelError):
        mm.download("m")
    part = tmp_path / "m" / "model.bin.part"
    assert part.exists() and 0 < part.stat().st_size < len(PAYLOAD)
    mm.download("m")  # resumes with a Range request
    assert mm.status("m")["hash_status"] == "pinned"
    assert (tmp_path / "m" / "model.bin").read_bytes() == PAYLOAD
    assert not part.exists()


def test_missing_model_paths_error(tmp_path, server):
    mm = ModelManager(tmp_path, [entry(server)])
    with pytest.raises(ModelError, match="not installed"):
        mm.paths("m")


def test_insufficient_disk(tmp_path, server):
    mm = ModelManager(tmp_path, [entry(server, size=10**15)])
    mm.accept_license("m")
    with pytest.raises(ModelError, match="disk space"):
        mm.download("m")


def test_shipped_manifest_is_consistent():
    from twod2vr180.models import load_manifest

    ms = load_manifest()
    ids = [m.id for m in ms]
    assert len(ids) == len(set(ids))
    for m in ms:
        assert m.license and m.license_url.startswith("https://")
        assert m.redistributable is False or m.license.startswith("Apache-2.0")
        for f in m.files:
            assert f.url.startswith("https://")

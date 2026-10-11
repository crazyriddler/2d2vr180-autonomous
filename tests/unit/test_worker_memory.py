"""Memory settings of the GPU workers (rc51, after an out-of-RAM failure of Qwen on a 32 GB PC)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workers"))
import _protocol  # noqa: E402
import qwen_views_worker as qw  # noqa: E402


def test_qwen_streams_its_blocks_and_never_uses_the_crashing_sync_mode():
    assert qw.memory_attempts(16376, 1.0, 31.9) == [("stream", 1.0), ("stream", 0.6)]
    assert qw.memory_attempts(24576, 1.0, 64.0)[0] == ("gpu", 1.0)


def test_pinned_cache_is_released():
    calls = []

    class Cuda:
        def synchronize(self):
            calls.append("sync")

        class memory:
            @staticmethod
            def host_memory_stats():
                return {"reserved_bytes.all.current": 3 * 2 ** 20}

    torch = type("T", (), {})()
    torch.cuda = Cuda()
    torch._C = type("C", (), {"_host_emptyCache": staticmethod(lambda: calls.append("empty"))})()
    assert qw.release_pinned(torch) == 3 and calls == ["sync", "empty"]
    assert qw.release_pinned(object()) is None          # builds without these APIs: no-op


def test_system_ram_is_read():
    total, free = qw.system_ram_gib()
    assert total is None or (total > 0 and 0 <= free <= total)


class FakeCuda:
    def __init__(self, free, total):
        self.info, self.frac = (free, total), None

    def mem_get_info(self):
        return self.info

    def set_per_process_memory_fraction(self, f):
        self.frac = f


def test_vram_cap_follows_the_free_memory(capsys):
    torch = type("T", (), {})()
    gib = 2 ** 30
    torch.cuda = FakeCuda(int(14.45 * gib), 16 * gib)          # the desktop and other apps use 1.5 GB
    f = _protocol.cap_vram(torch, 0.92)
    assert torch.cuda.frac == f and abs(f * 16 - (14.45 - 600 / 1024)) < 1e-6
    torch.cuda = FakeCuda(int(15.9 * gib), 16 * gib)
    assert _protocol.cap_vram(torch, 0.92) == 0.92
    assert "GPU memory" in capsys.readouterr().out


def test_ram_exhaustion_is_reported_as_out_of_memory(monkeypatch, capsys, tmp_path):
    import json

    req = tmp_path / "r.json"
    req.write_text("{}")
    monkeypatch.setattr(sys, "argv", ["w", str(req)])

    def main(r):
        raise MemoryError("Unable to allocate 11.9 MiB")

    try:
        _protocol.run(main)
    except SystemExit:
        pass
    ev = [json.loads(line[len(_protocol.PREFIX):]) for line in capsys.readouterr().out.splitlines()
          if line.startswith(_protocol.PREFIX)]
    assert ev[-1]["code"] == "oom" and "system memory (RAM)" in ev[-1]["message"]

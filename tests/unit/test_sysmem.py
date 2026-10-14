from twod2vr.sysmem import MemoryStatus, low_memory_warning, memory_status, parse_processes


def test_memory_status_reads_something():
    st = memory_status()
    assert st is None or (st.ram_total_gib > 0 and st.commit_limit_gib >= st.ram_total_gib * 0.5)


def test_low_memory_warning():
    assert low_memory_warning(MemoryStatus(31.9, 20, 40, 30)) is None
    w = low_memory_warning(MemoryStatus(31.9, 4, 36, 9.5))
    assert "9.5 GB" in w and "page file" in w
    assert low_memory_warning(None) is None


def test_parse_processes_from_powershell_json():
    one = '{"ProcessId":4242,"WorkingSetSize":16106127360,"ExecutablePath":"C:\\\\x\\\\runtimes\\\\gen\\\\python.exe"}'
    assert parse_processes(one) == [{"pid": 4242, "mib": 15360, "exe": "C:\\x\\runtimes\\gen\\python.exe"}]
    many = "[" + one + "," + one.replace("4242", "7") + "]"
    assert [p["pid"] for p in parse_processes(many)] == [4242, 7]
    assert parse_processes("") == [] and parse_processes("not json") == []

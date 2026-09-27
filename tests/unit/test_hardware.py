from twod2vr180.hardware import format_report, parse_nvidia_smi_csv, probe


def test_parse_nvidia_smi():
    gpus = parse_nvidia_smi_csv("0, NVIDIA GeForce RTX 4080, 16376, 15210, 576.02, 8.9\n"
                                "1, NVIDIA GeForce GTX 1060 6GB, 6144, 6000, 576.02, [N/A]\n")
    assert len(gpus) == 2
    assert gpus[0].is_target and gpus[0].total_mib == 16376 and gpus[0].compute_capability == "8.9"
    assert not gpus[1].is_target and gpus[1].compute_capability is None


def test_probe_runs_without_gpu():
    r = probe()
    txt = format_report(r)
    assert "CPU" in txt and r.cpu_count >= 1
    d = r.to_dict()
    assert "target_gpu_detected" in d

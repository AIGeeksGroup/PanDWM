from dwm_pano.datasets.horizondrive_pano import common_clip_indices


def _info(token, timestamp):
    return {"token": token, "timestamp": timestamp}


def test_common_clip_indices_uses_timestamp_sampling():
    # 12 Hz-ish source sampled at 10 Hz: index skipping would incorrectly keep
    # every source frame, while timestamp matching deliberately skips one.
    times = [0, 83_333, 166_667, 250_000, 333_333, 416_667]
    infos = [_info("t{}".format(i), ts) for i, ts in enumerate(times)]
    clips, audits = common_clip_indices(
        infos, [[item["token"] for item in infos]],
        sequence_length=4, fps=10, stride_s=1)
    assert clips == [[0, 1, 2, 4]]
    assert audits[0]["fps"] == 10.0


def test_common_clip_indices_rejects_duplicate_scene_membership():
    infos = [_info("a", 0), _info("b", 100_000)]
    try:
        common_clip_indices(
            infos, [["a", "b"], ["b"]], sequence_length=1,
            fps=10, stride_s=1)
    except ValueError as error:
        assert "duplicate token" in str(error)
    else:
        raise AssertionError("duplicate scene membership was accepted")

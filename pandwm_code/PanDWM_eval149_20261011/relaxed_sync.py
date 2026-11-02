"""Admit scene-0018's scene-start clip into the 150-scene val protocol (2026-10-19).

The frozen loader (dwm_pano.datasets.nuscenes_pano._enumerate_pano_clips) rejects a clip when any 10 Hz target time is
more than 50 ms (half a frame) from the nearest 12 Hz frame.  Of the 150 nuScenes val scenes only scene-0018 fails at
its scene start: one of 35 frames is 50.49 ms off (the worst accepted scene-start clip is 49.53 ms off), because its
interpolated frames are 108.5 ms apart for 0.6 s.  This patch accepts clips that fail ONLY on that check and are at
most LIMIT_US off; clips are still required to be monotonic and duplicate-free, and every other clip is unchanged.

  python3 relaxed_sync.py <script.py> <its arguments...>     (e.g. lowmem_launch.py run_clips_videos.py ...)
  import relaxed_sync; relaxed_sync.install()                 (before the dataset is built)
"""
import runpy
import sys

LIMIT_US = 51_000


def install(limit_us=LIMIT_US):
    import dwm_pano.datasets.nuscenes_pano as npano
    if getattr(npano._enumerate_pano_clips, "_relaxed", False):
        return
    original = npano._enumerate_pano_clips

    def enumerate_relaxed(scene_infos, sequence_length, fps_stride_tuples, emit_failures=False,
                          bad_timestamps_us=None):
        for clip, meta in original(scene_infos, sequence_length, fps_stride_tuples, emit_failures=True,
                                   bad_timestamps_us=bad_timestamps_us):
            if clip is not None:
                yield clip, meta
                continue
            idx = meta.get("indices") or []
            if meta.get("reason") == "max_time_error_exceeds_tolerance" and meta["max_time_error_us"] <= limit_us \
                    and len(set(idx)) == len(idx) == sequence_length and all(a < b for a, b in zip(idx, idx[1:])):
                meta = dict(meta, valid=True, relaxed_tolerance_us=limit_us)
                meta.pop("reason")
                yield [scene_infos[meta["scene_token"]][i] for i in idx], meta
            elif emit_failures:
                yield clip, meta

    enumerate_relaxed._relaxed = True
    npano._enumerate_pano_clips = enumerate_relaxed
    print(f"RELAXED_SYNC installed: time-error limit {limit_us} us for otherwise valid clips", flush=True)


if __name__ == "__main__":
    install()
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(script, run_name="__main__")

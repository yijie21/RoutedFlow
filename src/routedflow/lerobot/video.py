"""Lossless mp4 encode/decode helpers (libx264rgb qp0, GOP-bounded random access).

Codec decision (grill 2026-08-04): libx264rgb qp=0 is BITWISE lossless in RGB
(verified: encode->decode == original uint8), unlike the LeRobot-default AV1
yuv420p which is lossy and would perturb DINO inputs / policy pixels. GOP g=10
bounds random-access cost: seeking decodes at most g frames from the previous
keyframe (default g=250 would decode from frame 0 on every window read).
"""
import av
import numpy as np

FPS = 20
GOP = 10


def encode_video(frames_thwc_u8, path, fps=FPS, gop=GOP):
    """frames: (T, H, W, 3) uint8 -> lossless mp4 at path."""
    T, H, W, _ = frames_thwc_u8.shape
    out = av.open(str(path), "w")
    st = out.add_stream("libx264rgb", rate=fps)
    st.width, st.height, st.pix_fmt = W, H, "rgb24"
    st.options = {"qp": "0", "preset": "fast", "g": str(gop), "bf": "0"}
    for f in frames_thwc_u8:
        for pkt in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format="rgb24")):
            out.mux(pkt)
    for pkt in st.encode():
        out.mux(pkt)
    out.close()


_CONTAINERS = {}          # per-process (thus per-DataLoader-worker) LRU container cache
_CONTAINERS_MAX = 128     # bounded: each open codec holds decoder threads/buffers —
                          # an unbounded cache exhausts pthreads (EAGAIN) at ~700 open
                          # AUTO-threaded decoders (hit 2026-08-04)


def _container(path):
    path = str(path)
    hit = _CONTAINERS.pop(path, None)
    if hit is None:
        if len(_CONTAINERS) >= _CONTAINERS_MAX:
            oldest = next(iter(_CONTAINERS))     # insertion order + re-insert on hit = LRU
            _CONTAINERS.pop(oldest)[0].close()
        inp = av.open(path)
        st = inp.streams.video[0]
        st.codec_context.thread_count = 2  # bounded threads; ~20% slower than AUTO
        hit = (inp, st)
    _CONTAINERS[path] = hit
    return hit


def decode_window(path, start, count, fps=FPS):
    """Decode frames [start, start+count) -> (count, H, W, 3) uint8.
    Seeks to the enclosing keyframe, decodes forward; containers are kept open
    per process (~2 ms/open saved on the streaming hot path)."""
    inp, st = _container(path)
    inp.seek(int(start / fps / st.time_base), stream=st)
    # threaded decoder keeps in-flight frames across seeks -> BlockingIOError
    # on reformat without this flush (500-cycle soak test passes with it)
    st.codec_context.flush_buffers()
    frames = []
    for fr in inp.decode(video=0):
        idx = int(round(fr.pts * st.time_base * fps))
        if idx < start:
            continue
        frames.append(fr.to_ndarray(format="rgb24"))
        if len(frames) == count:
            break
    assert len(frames) == count, f"{path}: wanted {count} frames at {start}, got {len(frames)}"
    return np.stack(frames)


def decode_frame(path, index, fps=FPS):
    return decode_window(path, index, 1, fps)[0]


def decode_all(path):
    inp = av.open(str(path))
    frames = [fr.to_ndarray(format="rgb24") for fr in inp.decode(video=0)]
    inp.close()
    return np.stack(frames)

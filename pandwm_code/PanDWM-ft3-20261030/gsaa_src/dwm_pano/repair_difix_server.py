"""difix_ref worker process (2026-10-30, GS_NEARFIX_PLAN_20261030.md step 2).

Difix3D's pipeline and custom UNet / VAE need the versions its requirements.txt pins (diffusers 0.25.1,
transformers 4.38.0, huggingface-hub 0.25.1, peft 0.9.0); the evaluation environment has diffusers 0.31 (SD3).
So repair_difix.DifixRepair runs this file as a separate interpreter with the pinned packages first on PYTHONPATH,
and talks to it over stdin / stdout: every message is an 8-byte little-endian length plus a pickle.
Request {"image": (3, H, W) float16 in [0, 1], "ref": same} -> reply {"image": (3, H, W) float16}; {"quit": True}
or EOF ends the worker.  The first reply is {"ready": True, "versions": {...}}.  Does not import dwm_pano.

    python repair_difix_server.py <difix_ref model dir> <Difix3D/src>
"""

import pickle
import struct
import sys

PROMPT = "remove degradation"


def read_msg(f):
    head = f.read(8)
    if len(head) < 8:
        return None
    n = struct.unpack("<Q", head)[0]
    data = b""
    while len(data) < n:
        chunk = f.read(n - len(data))
        if not chunk:
            return None
        data += chunk
    return pickle.loads(data)


def write_msg(f, obj):
    data = pickle.dumps(obj, protocol=4)
    f.write(struct.pack("<Q", len(data)))
    f.write(data)
    f.flush()


def main():
    model_dir, src_dir = sys.argv[1], sys.argv[2]
    out = sys.stdout.buffer
    sys.stdout = sys.stderr                      # library prints must not corrupt the protocol stream
    import numpy as np
    import torch
    import diffusers
    import transformers
    sys.path.insert(0, src_dir)
    from pipeline_difix import DifixPipeline
    pipe = DifixPipeline.from_pretrained(model_dir, trust_remote_code=True)
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    write_msg(out, {"ready": True, "versions": {"torch": torch.__version__, "diffusers": diffusers.__version__,
                                                "transformers": transformers.__version__}})
    inp = sys.stdin.buffer
    while True:
        msg = read_msg(inp)
        if msg is None or msg.get("quit"):
            break
        img = torch.from_numpy(np.asarray(msg["image"], dtype=np.float32))[None].cuda()
        ref = torch.from_numpy(np.asarray(msg["ref"], dtype=np.float32))[None].cuda()
        with torch.no_grad():
            res = pipe(PROMPT, image=img, ref_image=ref, num_inference_steps=1, timesteps=[199],
                       guidance_scale=0.0, output_type="pt").images
        res = torch.as_tensor(res)
        if res.dim() == 4:
            res = res[0]
        if res.shape[0] != 3 and res.shape[-1] == 3:
            res = res.permute(2, 0, 1)
        write_msg(out, {"image": res.float().clamp(0, 1).cpu().numpy().astype(np.float16)})


if __name__ == "__main__":
    main()

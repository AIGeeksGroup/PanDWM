"""Distributed entry point used by pandwm_common_front.sbatch."""
import argparse
import sys
import types


def install_hardseam_guard():
    omninwm = types.ModuleType("omninwm")
    omninwm.__path__ = []
    utils = types.ModuleType("omninwm.utils")
    utils.__path__ = []
    hardseam = types.ModuleType("omninwm.utils.pano_hardseam")

    def unavailable(*args, **kwargs):
        raise RuntimeError(
            "online pano stitching unavailable; pre-rendered PNG required")

    hardseam.stitch_hardseam = unavailable
    omninwm.utils = utils
    utils.pano_hardseam = hardseam
    sys.modules.update({
        "omninwm": omninwm,
        "omninwm.utils": utils,
        "omninwm.utils.pano_hardseam": hardseam,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    install_hardseam_guard()
    from dwm_pano import evaluate_4step
    from dwm_pano.pipelines import ctsd_pano_consistency_distillation as cd

    original = cd.PanoCrossviewTemporalConsistencyDistillation \
        ._build_auxiliary_model

    def auxiliary(self, checkpoint_path, model_load_state_args,
                  requires_grad, role):
        if role == "teacher" and hasattr(self, "model_wrapper"):
            print("eval shim: aliasing consistency teacher to main model",
                  flush=True)
            return self.model, self.model_wrapper
        return original(
            self, checkpoint_path, model_load_state_args, requires_grad, role)

    cd.PanoCrossviewTemporalConsistencyDistillation._build_auxiliary_model = \
        auxiliary
    sys.argv = [
        "evaluate_4step.py", "-c", args.config, "-r", args.run,
        "-s", "10000", "--ema",
    ]
    evaluate_4step.main()


if __name__ == "__main__":
    main()

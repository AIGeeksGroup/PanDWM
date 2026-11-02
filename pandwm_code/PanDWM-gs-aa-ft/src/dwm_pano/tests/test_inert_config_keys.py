"""`INERT_OFFICIAL_STAGE2_KEYS` must keep matching the code, not a comment.

The Stage-2 configs name a distribution-matching objective, a critic, its
learning rate and betas, and a `teacher_forcing` flag. None of it runs: the
pipeline records those parameters in a bookkeeping dict and implements plain
consistency distillation. That is worth a runtime warning, but a warning based
on a hand-maintained list rots the moment someone wires one of the keys up --
and it would then be actively lying about the training objective.

So the set is re-derived here from the source, the same way it was produced:
a constructor parameter is inert when it has no load outside the bookkeeping
dict *and* `self.<name>` is never read in the module. Wiring a key up makes
this test fail, which is the point.
"""
import ast
import collections
import pathlib

from dwm_pano.pipelines.stage2_inert_keys import (
    INERT_OFFICIAL_STAGE2_KEYS, warn_inert_official_stage2_keys)

_MODULE = pathlib.Path(
    __file__).resolve().parents[1] \
    / "pipelines" / "ctsd_pano_consistency_distillation.py"


def _analyse():
    tree = ast.parse(_MODULE.read_text(encoding="utf-8"))
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    init = next(
        n for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "__init__")

    book = None
    for node in ast.walk(init):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Attribute)
                and t.attr == "official_stage2_parameters"
                for t in node.targets):
            book = node.value
    assert book is not None, "bookkeeping dict not found"
    inside = {id(n) for n in ast.walk(book)}

    local = collections.Counter(
        n.id for n in ast.walk(init)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
        and id(n) not in inside)
    attr = collections.Counter(
        n.attr for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load)
        and isinstance(n.value, ast.Name) and n.value.id == "self")

    keys = [k.value for k in book.keys]
    # A parameter whose only load is the right-hand side of `self.x = x`, with
    # `self.x` never read, is inert too -- that is exactly how the already
    # documented dead `timestep_shift` knob looks.
    derived_inert = {
        k for k in keys if attr.get(k, 0) == 0 and local.get(k, 0) <= 1
    }
    derived_live = set(keys) - derived_inert
    return set(keys), derived_inert, derived_live


def test_inert_set_matches_the_source():
    keys, derived_inert, derived_live = _analyse()

    newly_wired = sorted(INERT_OFFICIAL_STAGE2_KEYS & derived_live)
    assert not newly_wired, (
        "these keys are now read by the pipeline, so the warning would "
        "misdescribe the objective; drop them from "
        "INERT_OFFICIAL_STAGE2_KEYS: {}".format(newly_wired))

    newly_dead = sorted(derived_inert - INERT_OFFICIAL_STAGE2_KEYS)
    assert not newly_dead, (
        "these keys became inert and are not declared, so the warning would "
        "stay silent about them: {}".format(newly_dead))

    assert INERT_OFFICIAL_STAGE2_KEYS <= keys


def test_the_objective_misdescribing_keys_are_all_inert():
    """The whole reason the warning exists."""
    for key in (
            "distribution_loss", "dfake_gen_update_ratio", "lr_critic",
            "beta1_critic", "beta2_critic", "real_score_fsdp_wrap_strategy",
            "fake_score_fsdp_wrap_strategy", "teacher_forcing",
            "timestep_shift"):
        assert key in INERT_OFFICIAL_STAGE2_KEYS, key


def test_warning_only_reports_keys_the_config_set(capsys):
    warn_inert_official_stage2_keys(
        {"distribution_loss": None, "lr_critic": None, "seed": None})
    assert capsys.readouterr().out == ""

    warn_inert_official_stage2_keys(
        {"distribution_loss": "dmd", "lr_critic": 4e-07,
         "denoising_step_list": [1000, 750, 500, 250]})
    out = capsys.readouterr().out
    assert "plain consistency distillation" in out
    assert "distribution_loss='dmd'" in out
    assert "lr_critic" in out
    # A live key must never be reported as ignored.
    assert "denoising_step_list" not in out


def test_no_critic_is_built():
    """Independent of the key list: the pipeline has no critic at all."""
    source = _MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
    built = {
        n.targets[0].attr
        for n in ast.walk(cls)
        if isinstance(n, ast.Assign) and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Attribute)}
    for name in ("fake_score", "fake_score_wrapper", "critic_optimizer"):
        assert name not in built, \
            "a critic appeared; INERT_OFFICIAL_STAGE2_KEYS is now wrong"

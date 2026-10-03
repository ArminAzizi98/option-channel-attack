"""Defenses for a typed decision model used as a security gate.

Each defense is stated as a wrapper around the gate, so it can be deployed without
retraining and its cost is counted in extra model calls.

D1 label canonicalisation  - rewrite the option labels to A/B before the call, keeping the
                             definitions. Removes the option channel by construction.
D2 label-rename certificate- evaluate every label pair in the attacker's vocabulary. The
                             attack space is finite, so this is an exhaustive check rather
                             than a probabilistic certificate: if the decision is constant
                             over the vocabulary it cannot be changed by renaming.
D3 rendering ensemble      - average the distribution over M neutral renderings.
D4 disagreement gate       - escalate items where two renderings disagree.
D5 field-structured gating - pass the policy-relevant fields of the tool call rather than
                             the raw trace. The harness assembles the trace, so it already
                             holds the structured call; this removes the attacker's span
                             from the gate's input entirely.
D6 fail-closed default     - on abstention, block. The security-appropriate default, and the
                             one a confidence-gated deployment does not have.
"""
from typing import List, Dict, Any, Tuple, Callable, Optional
import numpy as np
from . import guardbench as G
from . import attacks as A

NEUTRAL_PAIRS = [("A", "B"), ("alpha", "beta"), ("opt1", "opt2"), ("X", "Y"), ("P", "Q")]

# which rendered trace lines each policy actually depends on
POLICY_FIELDS = {
    "exfil": ["destination"],
    "secret": ["body"],
    "escape": ["target path"],
    "exfil_secret": ["destination", "body"],
    "unauth_irrev": ["reversible", "authorization"],
    "escape_or_exfil": ["target path", "destination"],
}


def question_with_labels(policy_id: str, block_label: str, allow_label: str,
                         phrasing: str = "positive"):
    table = G.PHRASINGS[phrasing]
    _, block_def, allow_def, _, _ = next(p for p in table if p[0] == policy_id)
    return ({"type": "choice", "instructions": G.INSTRUCTIONS,
             "criteria": {allow_label: allow_def, block_label: block_def}},
            [allow_label, block_label])


# ----------------------------------------------------------------- D1 / D3 / D4
def defend_canonical(model, states, policy_id, phrasing="positive", batch_size=64):
    q, cn = question_with_labels(policy_id, "A", "B", phrasing)
    return model.score(states, q, cn, batch_size=batch_size)["probs"], 1


def defend_ensemble(model, states, policy_id, phrasing="positive", m=3, batch_size=64):
    acc = None
    for blk, alw in NEUTRAL_PAIRS[:m]:
        q, cn = question_with_labels(policy_id, blk, alw, phrasing)
        p = model.score(states, q, cn, batch_size=batch_size)["probs"]
        acc = p if acc is None else acc + p
    return acc / m, m


def defend_disagreement(model, states, policy_id, phrasing="positive", batch_size=64):
    """Returns (probs_from_first_rendering, flag, calls). Flagged items are escalated."""
    q1, cn1 = question_with_labels(policy_id, "block", "allow", phrasing)
    p1 = model.score(states, q1, cn1, batch_size=batch_size)["probs"]
    q2, cn2 = question_with_labels(policy_id, "A", "B", phrasing)
    p2 = model.score(states, q2, cn2, batch_size=batch_size)["probs"]
    return p1, (p1.argmax(1) != p2.argmax(1)), 2


# --------------------------------------------------------------------- D2
def certify_label_rename(model, states, policy_id, phrasing="positive", batch_size=64):
    """Exhaustive check over the attacker's label vocabulary.

    Returns per-item: the set of distinct decisions seen, and whether the decision is
    invariant across every label pair the attacker could install.
    """
    preds = []
    for blk, alw in A.option_channel_candidates():
        q, cn = question_with_labels(policy_id, blk, alw, phrasing)
        p = model.score(states, q, cn, batch_size=batch_size)["probs"]
        preds.append(p.argmax(1))
    P = np.stack(preds)                       # (n_renamings, n_items)
    invariant = (P == P[0]).all(axis=0)
    # worst case for the defender: does ANY renaming allow this item?
    any_allow = (P == 0).any(axis=0)
    return {"invariant": invariant, "any_allow": any_allow,
            "n_renamings": P.shape[0], "preds": P}


# --------------------------------------------------------------------- D5
def field_view(item, policy_id: str) -> str:
    """Render only the structured fields the policy references.

    The attacker-controlled tool output is not among them, so this defense removes the
    state channel as well as reducing distraction.
    """
    keep = POLICY_FIELDS[policy_id]
    lines = []
    for ln in item.state.split("\n"):
        stripped = ln.strip()
        for k in keep:
            if stripped.startswith(k + ":"):
                lines.append("  " + stripped)
                break
    return "\n".join(lines)


def defend_field_structured(model, items, policy_id, phrasing="positive",
                            block_label="A", allow_label="B", batch_size=64):
    states = [field_view(it, policy_id) for it in items]
    q, cn = question_with_labels(policy_id, block_label, allow_label, phrasing)
    return model.score(states, q, cn, batch_size=batch_size)["probs"], 1


# --------------------------------------------------------------------- D6
def apply_fail_closed(probs, flag):
    """Force BLOCK wherever the gate was flagged for escalation."""
    out = probs.copy()
    out[flag] = np.array([0.0, 1.0])
    return out

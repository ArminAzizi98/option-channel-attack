"""Self-contained model wrappers for the guardrail study.

Deliberately independent of the other project's code so neither can break the other.
Every wrapper returns probabilities in class order [ALLOW, BLOCK] plus whatever
uncertainty signal the model exposes.
"""
from typing import List, Dict, Any, Optional
import os, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HOME", f"{ROOT}/hf")
import numpy as np
from .config import ROOT


class LayaGate:
    kind = "laya"

    def __init__(self, model_id="convaiinnovations/laya-typed-decisions", device="cuda",
                 max_len=512, head_max_len=192):
        import laya
        self.model_id = model_id
        self.agent = laya.load(model_id, device=device)
        self.max_len, self.head_max_len = max_len, head_max_len

    def score(self, states: List[str], qspec: Dict[str, Any], class_names: List[str],
              batch_size: int = 64) -> Dict[str, np.ndarray]:
        probs, conf = [], []
        for i in range(0, len(states), batch_size):
            chunk = states[i:i + batch_size]
            res = self.agent.predict_batch(chunk, {"q": qspec}, batch_size=len(chunk),
                                           max_len=self.max_len, head_max_len=self.head_max_len)
            for r in res:
                a = r["answers"]["q"]
                pr = a["probabilities"]
                probs.append([float(pr[class_names[0]]), float(pr[class_names[1]])])
                conf.append(float(a.get("confidence", np.nan)))
        return {"probs": np.array(probs), "confidence": np.array(conf)}


class VonGate:
    kind = "von"

    def __init__(self, model_id="von-latest", device="cuda", **kw):
        import von
        self.von = von
        self.model_id = model_id

    def score(self, states, qspec, class_names, batch_size: int = 1):
        probs, conf = [], []
        q = self.von.Choice(instructions=qspec["instructions"], criteria=qspec["criteria"])
        for s in states:
            r = self.von.system_one(s, {"q": q})
            a = r.answers["q"]
            pr = a.probabilities
            probs.append([float(pr[class_names[0]]), float(pr[class_names[1]])])
            conf.append(float(getattr(a, "confidence", np.nan)))
        return {"probs": np.array(probs), "confidence": np.array(conf)}


class LMGate:
    """Ordinary LM label log-probability readout, the baseline the audit found missing."""
    kind = "lm"

    def __init__(self, model_id="Qwen/Qwen2.5-1.5B-Instruct", device="cuda", dtype="bfloat16",
                 max_chars: int = 6000):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=getattr(torch, dtype), device_map=device)
        self.model.eval()
        self.device = device
        self.max_chars = max_chars

    def _prompt(self, state, qspec, names):
        crit = qspec["criteria"]
        lines = [f"Agent trace:\n{state[:self.max_chars]}\n",
                 f"Question: {qspec['instructions']}", "Options:"]
        for nm in names:
            lines.append(f"- {nm}: {crit[nm]}")
        lines.append("Answer:")
        return "\n".join(lines)

    def score(self, states, qspec, class_names, batch_size: int = 1):
        import torch
        probs = []
        for s in states:
            prompt = self._prompt(s, qspec, class_names)
            msgs = [{"role": "user", "content": prompt}]
            base = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            base_ids = self.tok(base, return_tensors="pt", add_special_tokens=False).input_ids
            lps = []
            for nm in class_names:
                cont = self.tok(" " + nm, return_tensors="pt", add_special_tokens=False).input_ids
                ids = torch.cat([base_ids, cont], dim=1).to(self.device)
                with torch.no_grad():
                    logits = self.model(ids).logits.float()
                lsm = torch.log_softmax(logits[0, :-1], dim=-1)
                tgt = ids[0, 1:]
                n = cont.shape[1]
                lps.append(lsm[-n:].gather(-1, tgt[-n:].unsqueeze(-1)).sum().item() / n)
            a = np.array(lps) - max(lps)
            p = np.exp(a); p = p / p.sum()
            probs.append(p)
        probs = np.array(probs)
        return {"probs": probs, "confidence": np.full(len(probs), np.nan)}


class RLCDGate:
    """`rlcd` / Verdict-open-jev: a GLiClass encoder with an explicit abstention option.

    Its `Option(id, description)` is exactly the label/definition split this paper is about,
    and unlike every other model here it can return "insufficient evidence" instead of a
    decision. We renormalise over the two real options so the numbers are comparable with
    the other gates, and keep the abstention mass separately, since a gate that abstains
    under attack is behaving correctly rather than failing.
    """
    kind = "rlcd"

    def __init__(self, model_id="heman10x/rlcd-modernbert-151m", device="cuda", **kw):
        import rlcd
        self.rlcd = rlcd
        self.model_id = model_id
        self.engine = rlcd.DecisionEngine(model_name_or_path=model_id, device=device)

    def score(self, states, qspec, class_names, batch_size: int = 1):
        crit = qspec["criteria"]
        opts = tuple(self.rlcd.Option(id=nm, description=crit[nm]) for nm in class_names)
        q = self.rlcd.Choice(id="gate", question=qspec["instructions"], options=opts)
        probs, conf, abst = [], [], []
        for st in states:
            r = self.engine.evaluate(context=st, queries=[q]).results[0]
            pr = r.probabilities
            a = float(pr.get(self.rlcd.INSUFFICIENT_EVIDENCE_ID, 0.0))
            p0 = float(pr.get(class_names[0], 0.0)); p1 = float(pr.get(class_names[1], 0.0))
            tot = p0 + p1
            probs.append([p0 / tot, p1 / tot] if tot > 0 else [0.5, 0.5])
            abst.append(a); conf.append(1.0 - a)
        return {"probs": np.array(probs), "confidence": np.array(conf),
                "p_abstain": np.array(abst)}


REGISTRY = {
    "laya-td": lambda d="cuda": LayaGate("convaiinnovations/laya-typed-decisions", device=d),
    "laya-en": lambda d="cuda": LayaGate("convaiinnovations/laya", device=d),
    "laya-ml": lambda d="cuda": LayaGate("convaiinnovations/laya-multilingual", device=d),
    "von":     lambda d="cuda": VonGate(device=d),
    "rlcd":    lambda d="cuda": RLCDGate(device=d),
    "qwen1.5b": lambda d="cuda": LMGate("Qwen/Qwen2.5-1.5B-Instruct", device=d),
    "qwen7b":   lambda d="cuda": LMGate("Qwen/Qwen2.5-7B-Instruct", device=d),
}


def load(name, device="cuda"):
    return REGISTRY[name](device)

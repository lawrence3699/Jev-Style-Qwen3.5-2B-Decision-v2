from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import unicodedata

import numpy as np
from scipy.optimize import minimize_scalar

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
HEADER = "You are a decision function. Read the state, then answer the question by choosing exactly one option."
CORE = ["sst2", "mnli", "ag_news", "boolq", "sst5"]


def sha_file(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized(text):
    return " ".join(re.findall(r"\w+", unicodedata.normalize("NFKC", str(text)).casefold()))


def read_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def content_groups(state, components=()):
    """Exact/component and legacy 140-whitespace-word clip fingerprints."""
    def clipped(s):
        return " ".join(str(s).split()[:140])
    texts = [state, clipped(state)]
    for component in components:
        if len(normalized(component).split()) >= 5:
            texts += [component, clipped(component)]
    if str(state).startswith("Premise: ") and "\nHypothesis: " in state:
        premise, hypothesis = state[9:].split("\nHypothesis: ", 1)
        texts.append(f"Premise: {clipped(premise)}\nHypothesis: {clipped(hypothesis)}")
    groups = {digest(normalized(t)) for t in texts if normalized(t)}
    return sorted(groups or {digest("raw:" + str(state).strip())})


def record(task, state, question, options, label, source, kind="choice", target=None,
           components=(), group=None, **extra):
    options = [str(o) for o in options]
    if not str(state).strip() or not 2 <= len(options) <= 26 or len(set(options)) != len(options):
        raise ValueError("invalid state or options")
    if not 0 <= int(label) < len(options):
        raise ValueError("invalid label")
    if target is None:
        target = [float(i == label) for i in range(len(options))]
    target = np.asarray(target, dtype=float)
    if len(target) != len(options) or not np.all(np.isfinite(target)) or np.any(target < 0) or target.sum() <= 0:
        raise ValueError("invalid target distribution")
    if abs(float(target.sum()) - 1) > 1e-4:
        raise ValueError("target does not sum to one")
    target = (target / target.sum()).tolist()
    state, question = str(state), str(question)
    groups = content_groups(state, components)
    if group:
        groups.append(digest("source-group:" + str(group)))
    uid = digest(json.dumps([task, normalized(state), question, options], ensure_ascii=False))
    return dict(uid=uid, task=task, kind=kind, state=state, question=question, options=options,
                label=int(label), target=target, source=source, group_ids=sorted(set(groups)),
                cluster_id=digest(str(group)) if group else groups[0], **extra)


def render(row, order=None, template=0):
    order = list(range(len(row["options"]))) if order is None else list(order)
    if sorted(order) != list(range(len(row["options"]))):
        raise ValueError("not a permutation")
    if row["kind"] == "score" and order != list(range(len(order))):
        raise ValueError("score levels cannot be reordered")
    options = [row["options"][i] for i in order]
    lines = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    if template == 0:
        prompt = f"{HEADER}\n\n[State]\n{row['state']}\n\n[Question]\n{row['question']}\n\n[Options]\n{lines}\n\nAnswer:"
    elif template == 1:
        prompt = f"Read the information and select the single best answer from the listed options.\n\nInformation:\n{row['state']}\n\nQuestion: {row['question']}\n\nOptions:\n{lines}\n\nAnswer:"
    else:
        raise ValueError("unknown template")
    return {**row, "prompt": prompt, "options": options, "label": order.index(row["label"]),
            "target": [row["target"][i] for i in order], "option_order": order,
            "template": template}


def view(row, rng=None, augment=False):
    order = list(range(len(row["options"])))
    if augment and row["kind"] != "score":
        rng.shuffle(order)
    template = 1 if augment and rng.random() < 0.2 else 0
    return render(row, order, template)


def audit_splits(splits):
    groups = {role: {g for row in rows for g in row["group_ids"]} for role, rows in splits.items()}
    result = {}
    roles = list(groups)
    for i, a in enumerate(roles):
        for b in roles[i+1:]:
            n = len(groups[a] & groups[b])
            result[f"{a}__{b}"] = n
            if n:
                raise ValueError(f"content leakage between {a} and {b}: {n} groups")
    return result


def probabilities(logits, temperature=1.0):
    z = np.asarray(logits, dtype=np.float64) / float(temperature)
    if not np.all(np.isfinite(z)) or temperature <= 0:
        raise ValueError("nonfinite logits or invalid temperature")
    exp = np.exp(z - np.max(z))
    return exp / exp.sum()


def fit_temperature(predictions):
    if not predictions:
        raise ValueError("empty calibration set")
    # Same sample-weighted proper log loss and bounds for every model.
    def objective(log_t):
        t = math.exp(log_t)
        losses = []
        for r in predictions:
            z = np.asarray(r["logits"], dtype=float) / t
            z -= z.max()
            log_p = z - np.log(np.exp(z).sum())
            losses.append(-np.dot(r["target"], log_p))
        return float(np.mean(losses))
    result = minimize_scalar(objective, bounds=(math.log(.05), math.log(20)), method="bounded")
    if not result.success:
        raise RuntimeError("temperature optimization failed")
    return {"temperature": math.exp(result.x), "calibration_n": len(predictions),
            "objective": "sample_mean_soft_cross_entropy", "bounds": [.05, 20],
            "nll_before": objective(0), "nll_after": float(result.fun)}


def summarize(predictions, temperature=1.0):
    by_task = {}
    for task in sorted({r["task"] for r in predictions}):
        rows = [r for r in predictions if r["task"] == task]
        ps = [probabilities(r["logits"], temperature) for r in rows]
        labels = np.array([r["label"] for r in rows])
        pred = np.array([p.argmax() for p in ps])
        correct = pred == labels
        conf = np.array([p.max() for p in ps])
        soft = any(r.get("reference") == "teacher" for r in rows)
        ce, brier, mae = [], [], []
        for r, p in zip(rows, ps):
            q = np.asarray(r["target"], dtype=float)
            ce.append(float(-np.dot(q, np.log(np.clip(p, 1e-300, 1)))))
            brier.append(float(np.square(p-q).sum()))
            if r["kind"] == "score":
                levels = np.arange(len(p))
                mae.append(abs(float(np.dot(p-q, levels))) / (len(p)-1))
        ece = 0.0
        for lo, hi in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
            sel = (conf >= lo) & ((conf < hi) if hi < 1 else (conf <= hi))
            if sel.any():
                ece += float(sel.mean()) * abs(float(correct[sel].mean()-conf[sel].mean()))
        # Option indices are local aliases and may be permuted per example.
        gold_names = np.array([r["options"][r["label"]] for r in rows])
        pred_names = np.array([r["options"][int(i)] for r,i in zip(rows,pred)])
        f1s = []
        for label in sorted(set(gold_names.tolist()) | set(pred_names.tolist())):
            tp = int(((pred_names == label) & (gold_names == label)).sum())
            fp = int(((pred_names == label) & (gold_names != label)).sum())
            fn = int(((pred_names != label) & (gold_names == label)).sum())
            f1s.append(2*tp / max(1, 2*tp+fp+fn))
        by_task[task] = dict(n=len(rows), accuracy=float(correct.mean()), macro_f1=float(np.mean(f1s)),
                             nll=float(np.mean(ce)), brier=float(np.mean(brier)), ece=ece,
                             reference="teacher" if soft else "label", normalized_score_mae=float(np.mean(mae)) if mae else None)
    def macro(tasks):
        return {k: float(np.mean([by_task[t][k] for t in tasks])) for k in ["accuracy", "macro_f1", "nll", "brier", "ece"]} if tasks else {}
    return {"temperature": temperature, "by_task": by_task,
            "real_label_macro": macro([t for t in by_task if by_task[t]["reference"] == "label"]),
            "teacher_macro": macro([t for t in by_task if by_task[t]["reference"] == "teacher"]),
            "core_macro": macro([t for t in CORE if t in by_task])}

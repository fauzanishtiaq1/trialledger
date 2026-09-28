from __future__ import annotations

import csv, json, math, pathlib, random, itertools
from typing import Mapping, Sequence

_BINARY = {"accuracy", "log_loss", "brier", "roc_auc"}
_REG = {"mae", "rmse", "r2"}

def _finite(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))

def load_records(path: str | pathlib.Path, *, fmt: str = "auto", preserve_columns=("id", "group", "model")) -> list[dict[str, object]]:
    p = pathlib.Path(path); text = p.read_text(encoding="utf-8")
    if fmt == "auto": fmt = "csv" if p.suffix.lower() == ".csv" else "jsonl"
    if fmt == "csv":
        out=[]
        reader = csv.DictReader(text.splitlines())
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames) or any(not x for x in reader.fieldnames):
            raise ValueError("CSV headers must be nonempty and unique")
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError("CSV rows must match header width")
            converted={}
            for k,v in row.items():
                if k in preserve_columns:
                    converted[k] = v
                    continue
                if v is None: converted[k]=v; continue
                try: converted[k]=float(v) if any(c in v for c in ".eE") else int(v)
                except (ValueError, TypeError): converted[k]=v
            out.append(converted)
        return out
    if fmt != "jsonl": raise ValueError("fmt must be auto, csv, or jsonl")
    out=[]
    for i,line in enumerate(text.splitlines(), 1):
        if line.strip():
            v=json.loads(line)
            if not isinstance(v, dict): raise ValueError(f"line {i} is not an object")
            out.append(v)
    return out

def validate_records(records: Sequence[Mapping[str, object]], *, target="y_true", prediction_columns=None, task="binary", metric="roc_auc", id_column="id", fold_column="fold", group_column=None):
    errors=[]; warnings=[]; n=len(records); ids=[]; seen=set(); dup=[]
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)): errors.append("records must be a sequence")
    if n == 0: errors.append("records must not be empty")
    if any(not isinstance(r, Mapping) for r in records):
        errors.append("each record must be a mapping")
        return {"valid":False,"n_rows":n,"ids":{"unique":False,"missing":n,"duplicates":[]},"columns":{},"errors":errors,"warnings":warnings}
    if task not in ("binary", "regression"): errors.append("task must be binary or regression")
    if (task == "binary" and metric not in _BINARY) or (task == "regression" and metric not in _REG): errors.append("unsupported task/metric pair")
    for r in records:
        if not isinstance(r, Mapping):
            continue
        ident=r.get(id_column)
        if not isinstance(ident, str) or not ident.strip(): errors.append("missing or empty id")
        else:
            ids.append(ident)
            if ident in seen and ident not in dup: dup.append(ident)
            seen.add(ident)
    if dup: errors.append("duplicate ids")
    preds=list(prediction_columns or [])
    if len(set(preds)) != len(preds): errors.append("prediction columns must be unique")
    if not preds and records: preds=[k for k in records[0] if k not in (id_column,target,fold_column,group_column,"group")]
    cols={}
    if not preds: errors.append("at least two prediction columns are required")
    if len(preds)<2: errors.append("at least two prediction columns are required")
    labels=[]
    for r in records:
        y=r.get(target)
        if not _finite(y): errors.append(f"target {target} must be finite numeric")
        else: labels.append(float(y))
        if task=="binary" and _finite(y) and float(y) not in (0.0,1.0): errors.append("binary labels must be 0/1")
        for p in preds:
            v=r.get(p)
            if not _finite(v): errors.append(f"prediction {p} must be finite numeric")
            elif task=="binary" and metric in _BINARY and not 0<=float(v)<=1: errors.append(f"binary prediction {p} must be in [0,1]")
    if task=="binary" and labels and set(labels)!={0.0,1.0}: errors.append("binary target must contain both classes")
    if fold_column:
        for r in records:
            if fold_column in r and r[fold_column] is not None and (not isinstance(r[fold_column], int) or isinstance(r[fold_column], bool) or r[fold_column] < 0):
                errors.append(f"{fold_column} must be a non-negative integer")
    if group_column:
        for r in records:
            value = r.get(group_column)
            if value is not None and not isinstance(value, (str, int)):
                errors.append(f"{group_column} must contain string or integer identifiers")
    for p in preds: cols[p]={"present":sum(_finite(r.get(p)) for r in records), "finite":sum(_finite(r.get(p)) for r in records)}
    return {"valid":not errors,"n_rows":n,"ids":{"unique":not dup,"missing":sum(not isinstance(r.get(id_column),str) or not r.get(id_column) for r in records),"duplicates":dup},"columns":cols,"errors":list(dict.fromkeys(errors)),"warnings":warnings}

def _auc(y,p):
    npos=sum(z==1 for z in y); nneg=sum(z==0 for z in y)
    if not npos or not nneg: return None
    ordered=sorted(zip(p,y), key=lambda t:t[0]); rank_sum=0.0; i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and ordered[j][0]==ordered[i][0]: j+=1
        rank=(i+1+j)/2
        rank_sum += rank*sum(z==1 for _,z in ordered[i:j]); i=j
    return (rank_sum-npos*(npos+1)/2)/(npos*nneg)
def _score(y,p,task,metric):
    if task=="binary":
        if metric=="accuracy": return sum((x>=.5)==bool(z) for z,x in zip(y,p))/len(y)
        if metric=="roc_auc": return _auc(y,p)
        if metric=="brier": return -sum((x-z)**2 for z,x in zip(y,p))/len(y)
        return sum(math.log(max(1e-15, min(1-1e-15, x if z else 1-x))) for z,x in zip(y,p))/len(y)
    if metric=="mae": return -sum(abs(x-z) for z,x in zip(y,p))/len(y)
    if metric=="rmse": return -math.sqrt(sum((x-z)**2 for z,x in zip(y,p))/len(y))
    mean=sum(y)/len(y); den=sum((z-mean)**2 for z in y)
    return None if den==0 else 1-sum((z-x)**2 for z,x in zip(y,p))/den
def _quant(vals,q):
    if not vals:return None
    a=sorted(vals); pos=(len(a)-1)*q; lo=int(pos); hi=min(lo+1,len(a)-1); return a[lo]+(a[hi]-a[lo])*(pos-lo)

def compare_experiments(records, *, prediction_columns, target="y_true", task="binary", metric="roc_auc", id_column="id", fold_column="fold", group_column=None, bootstrap=2000, seed=0, alpha=.05, permutation=2000):
    if not isinstance(bootstrap,int) or isinstance(bootstrap,bool) or bootstrap<1: raise ValueError("bootstrap must be a positive integer")
    if not isinstance(permutation,int) or isinstance(permutation,bool) or permutation<1: raise ValueError("permutation must be a positive integer")
    if not isinstance(seed,int) or isinstance(seed,bool): raise ValueError("seed must be an integer")
    if not _finite(alpha) or not 0<alpha<1: raise ValueError("alpha must be finite and between 0 and 1")
    if len(prediction_columns)<2 or len(set(prediction_columns))!=len(prediction_columns): raise ValueError("at least two unique prediction columns are required")
    val=validate_records(records,target=target,prediction_columns=prediction_columns,task=task,metric=metric,id_column=id_column,fold_column=fold_column,group_column=group_column)
    if not val["valid"]: raise ValueError("invalid records: " + "; ".join(val["errors"]))
    names=list(prediction_columns); y=[float(r[target]) for r in records]; n=len(y)
    rng=random.Random(seed); arrays={name:[float(r[name]) for r in records] for name in names}; scores={name:_score(y,arrays[name],task,metric) for name in names}
    if any(v is None for v in scores.values()): raise ValueError("metric is undefined for the supplied data (for example, constant target for r2)")
    pair_boot={pair:[] for pair in itertools.combinations(names,2)}; model_boot={name:[] for name in names}
    for _ in range(bootstrap):
        ix=[rng.randrange(n) for _ in range(n)]; yy=[y[i] for i in ix]; vals={name:_score(yy,[arrays[name][i] for i in ix],task,metric) for name in names}
        if all(v is not None and math.isfinite(v) for v in vals.values()):
            for name,v in vals.items(): model_boot[name].append(v)
            for pair in pair_boot: pair_boot[pair].append(vals[pair[0]]-vals[pair[1]])
    pair_perm={pair:[] for pair in pair_boot}
    for _ in range(permutation):
        for pair in pair_perm:
            aa=[];bb=[]
            for x,z in zip(arrays[pair[0]],arrays[pair[1]]): (aa.append(z),bb.append(x)) if rng.getrandbits(1) else (aa.append(x),bb.append(z))
            x=_score(y,aa,task,metric); z=_score(y,bb,task,metric)
            if x is not None and z is not None: pair_perm[pair].append(x-z)
    diagnostics = {}
    for label, column in (("fold", fold_column), ("group", group_column)):
        if not column:
            continue
        counts = {}
        missing = 0
        for row in records:
            value = row.get(column)
            if value is None or value == "":
                missing += 1
            else:
                key = (type(value).__name__, value)
                counts[key] = counts.get(key, 0) + 1
        diagnostics[label] = {"column": column, "missing": missing,
            "unique": len(counts), "counts": [
                {"value": value, "rows": count} for (_, value), count in
                sorted(counts.items(), key=lambda item: (item[0][0], str(item[0][1])))]}
    lim=["paired row bootstrap; group identifiers are diagnostic only, not cluster robust", "multiple pairwise comparisons are not corrected"]
    for pair in pair_boot:
        if len(pair_boot[pair])<bootstrap: lim.append(f"invalid bootstrap draws omitted for {pair[0]}-{pair[1]}: {bootstrap-len(pair_boot[pair])}")
    models={name:{"score":scores[name],"raw_score":-scores[name] if metric in {"log_loss","brier","mae","rmse"} else scores[name],"ci_low":_quant(model_boot[name],alpha/2),"ci_high":_quant(model_boot[name],1-alpha/2)} for name in names}
    pairwise=[]
    for pair in pair_boot:
        d=scores[pair[0]]-scores[pair[1]]; pp=pair_perm[pair]; pairwise.append({"left":pair[0],"right":pair[1],"delta":d,"delta_ci_low":_quant(pair_boot[pair],alpha/2),"delta_ci_high":_quant(pair_boot[pair],1-alpha/2),"p_permutation":(1+sum(abs(x)>=abs(d) for x in pp))/(len(pp)+1) if pp else None,"n_pairs":n})
    return {"schema_version":"1","task":task,"metric":metric,"n_rows":n,"models":models,"pairwise":pairwise,"bootstrap":{"method":"paired_row_bootstrap","iterations":bootstrap,"valid_draws":len(next(iter(pair_boot.values()))),"seed":seed,"alpha":alpha},"permutation":{"iterations":permutation,"valid_draws":len(next(iter(pair_perm.values())))},"validation":val,"diagnostics":diagnostics,"limitations":lim}

def bootstrap_selection(fold_scores, *, model_column="model", fold_column="fold", score_column="score", bootstrap=2000, seed=0, alpha=.05):
    if not isinstance(bootstrap,int) or isinstance(bootstrap,bool) or bootstrap<1: raise ValueError("bootstrap must be a positive integer")
    if not isinstance(seed,int) or isinstance(seed,bool): raise ValueError("seed must be an integer")
    if not _finite(alpha) or not 0<alpha<1: raise ValueError("alpha must be finite and between 0 and 1")
    rows=list(fold_scores)
    if any(not isinstance(r,Mapping) for r in rows): raise ValueError("fold scores must be mappings")
    folds=sorted({r.get(fold_column) for r in rows}, key=lambda x:(type(x).__name__,repr(x))); models=sorted({r.get(model_column) for r in rows}, key=lambda x:(type(x).__name__,repr(x)))
    if any(f is None or f=="" for f in folds) or any(m is None or m=="" for m in models): raise ValueError("fold and model identifiers must be non-empty")
    if not folds or not models or any(not _finite(r.get(score_column)) for r in rows): raise ValueError("fold scores must be finite and non-empty")
    table={(r.get(fold_column),r.get(model_column)):float(r.get(score_column)) for r in rows}
    if len(table)!=len(rows) or any((f,m) not in table for f in folds for m in models): raise ValueError("all models must cover the same folds exactly once")
    rng=random.Random(seed); vals=[]; invalid=0
    for _ in range(bootstrap):
        bag=[folds[rng.randrange(len(folds))] for _ in folds]; oob=[f for f in folds if f not in bag]
        if not oob: invalid+=1; continue
        means={m:sum(table[f,m] for f in bag)/len(bag) for m in models}; winner=sorted(models,key=lambda m:(-means[m],str(m)))[0]
        vals.append(sum(table[f,winner] for f in oob)/len(oob))
    if len(vals)<max(100,bootstrap*.5): raise ValueError(f"too few valid bootstrap draws: {len(vals)}")
    return {"method":"bbc_f_style_fold_selection","point_estimate":sum(vals)/len(vals),"ci_low":_quant(vals,alpha/2),"ci_high":_quant(vals,1-alpha/2),"bootstrap":bootstrap,"valid_draws":len(vals),"invalid_draws":invalid,"seed":seed,"alpha":alpha,"n_folds":len(folds),"n_models":len(models),"limitations":["simplified fold-score implementation; no refitting or coverage guarantee"]}

def render_report(result, *, format="json"):
    if format=="json": return json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n"
    if format!="markdown": raise ValueError("format must be json or markdown")
    lines=["# TrialLedger report",f"- Task: {result.get('task')}",f"- Metric: {result.get('metric')}",f"- Rows: {result.get('n_rows')}","","## Pairwise results"]
    for p in result.get("pairwise",[]): lines.append(f"- {p['left']} vs {p['right']}: delta={p['delta']}, CI=({p['delta_ci_low']}, {p['delta_ci_high']}), permutation p={p['p_permutation']}")
    lines += ["","## Limitations"]+[f"- {x}" for x in result.get("limitations",[])]
    return "\n".join(lines)+"\n"

"""Measured CLI comparison against an independent paired-accuracy baseline."""
import argparse, csv, json, math, platform, random, statistics, subprocess, sys, tempfile, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def invoke(arguments):
    started=time.perf_counter()
    result=subprocess.run([sys.executable,*map(str,arguments)],cwd=ROOT,text=True,capture_output=True,encoding="utf-8")
    if result.returncode: raise RuntimeError(result.stderr)
    return time.perf_counter()-started

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--rows",type=int,default=100000)
    parser.add_argument("--iterations",type=int,default=200)
    parser.add_argument("--repetitions",type=int,default=3)
    args=parser.parse_args()
    if min(args.rows,args.iterations,args.repetitions)<1:parser.error("counts must be positive")
    runs=[]
    with tempfile.TemporaryDirectory(prefix="trialledger-benchmark-") as directory:
        work=Path(directory)
        rng=random.Random(17)
        rows=[]
        for i in range(args.rows):
            label=int(rng.random()<.4)
            rows.append({"id":str(i),"y_true":label,"a":label if rng.random()>=.08 else 1-label,"b":label if rng.random()>=.15 else 1-label})
        source=work/"predictions.csv"
        with source.open("w",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=["id","y_true","a","b"]);writer.writeheader();writer.writerows(rows)
        for repetition in range(args.repetitions):
            report_path=work/f"tool-{repetition}.json"; baseline_path=work/f"baseline-{repetition}.json";seed=500+repetition
            tool_time=invoke(["-m","trialledger","compare",source,"--pred","a","b","--metric","accuracy","--bootstrap",args.iterations,"--permutation",args.iterations,"--seed",seed,"--out",report_path])
            baseline_time=invoke(["benchmarks/baseline.py",source,"--iterations",args.iterations,"--seed",seed,"--output",baseline_path])
            result=json.loads(report_path.read_text(encoding="utf-8"));oracle=json.loads(baseline_path.read_text(encoding="utf-8"));pair=result["pairwise"][0]
            comparisons={"accuracy_a":result["models"]["a"]["score"],"accuracy_b":result["models"]["b"]["score"],"delta":pair["delta"],"ci_low":pair["delta_ci_low"],"ci_high":pair["delta_ci_high"],"p":pair["p_permutation"]}
            assert all(math.isclose(comparisons[key],value,abs_tol=1e-12) for key,value in oracle.items())
            assert .5 < oracle["accuracy_b"] < oracle["accuracy_a"] < 1
            runs.append({"seed":seed,"tool_seconds":tool_time,"baseline_seconds":baseline_time,"tool":comparisons,"independent_baseline":oracle})
        fold_values={"a":[.90,.55,.90,.55,.90],"b":[.75,.76,.77,.78,.79],"c":[.61,.62,.63,.64,.65]}
        fold_rows=[{"fold":i,"model":model,"score":score} for model,scores in fold_values.items() for i,score in enumerate(scores)]
        folds=work/"folds.jsonl";folds.write_text("".join(json.dumps(row)+"\n" for row in fold_rows),encoding="utf-8");selection=work/"selection.json"
        selection_time=invoke(["-m","trialledger","select",folds,"--bootstrap",500,"--seed",0,"--out",selection])
        result=json.loads(selection.read_text(encoding="utf-8"))
        # Independent fold bootstrap of a fixed literal score matrix.
        rng=random.Random(0);values=[]
        for _ in range(500):
            bag=[rng.randrange(5) for _ in range(5)];remaining=set(range(5))-set(bag)
            if not remaining:continue
            winner=min(fold_values,key=lambda model:(-sum(fold_values[model][j] for j in bag),model))
            values.append(sum(fold_values[winner][j] for j in sorted(remaining))/len(remaining))
        assert result["valid_draws"]==len(values) and math.isclose(result["point_estimate"],statistics.mean(values),abs_tol=1e-12)
        selection_result={"fold_scores":fold_values,"naive_best_mean":max(statistics.mean(v) for v in fold_values.values()),"selection_adjusted":result,"independent_mean":statistics.mean(values),"seconds":selection_time}
    report={"configuration":{"rows":args.rows,"iterations":args.iterations,"repetitions":args.repetitions,"fixture_seed":17,"metric":"accuracy"},"environment":{"python":platform.python_version(),"platform":platform.platform(),"machine":platform.machine()},"runs":runs,"median_tool_seconds":statistics.median(x["tool_seconds"] for x in runs),"median_baseline_seconds":statistics.median(x["baseline_seconds"] for x in runs),"fold_selection":selection_result,"limits":["Synthetic IID correctness and runtime benchmark, not human time saved.","Baseline is specialized to accuracy and omits general validation/reporting; a tool speedup is not promised.","Fold bootstrap is a selection-aware estimate, not a guarantee of external performance."]}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print("BENCHMARK_OK")

if __name__=="__main__":main()

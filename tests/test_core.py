import math
import unittest
import tempfile
from pathlib import Path
from trialledger.core import *

def rows():
    return [{"id":f"r{i}","y_true":y,"a":pa,"b":pb} for i,(y,pa,pb) in enumerate([(0,0.1,0.9),(0,0.2,0.8),(1,0.8,0.2),(1,0.9,0.1)])]

class CoreTests(unittest.TestCase):
 def test_validation_and_auc_ties(self):
    r=rows(); assert validate_records(r,prediction_columns=["a","b"],metric="roc_auc")["valid"]
    out=compare_experiments(r,prediction_columns=["a","b"],metric="roc_auc",bootstrap=100,permutation=100,seed=4)
    assert out["models"]["a"]["score"]==1 and out["models"]["b"]["score"]==0
    assert out["pairwise"][0]["delta"]==1

 def test_invalid_duplicate_and_binary(self):
    r=rows(); r[1]["id"]="r0"
    v=validate_records(r,prediction_columns=["a","b"]); assert not v["valid"] and v["ids"]["duplicates"]==["r0"]
    r=rows(); r[0]["a"]=2; assert not validate_records(r,prediction_columns=["a","b"])["valid"]

 def test_regression_and_reproducible(self):
    r=[{"id":str(i),"y_true":y,"a":p,"b":q} for i,(y,p,q) in enumerate([(1,1,2),(2,2,2),(3,4,3)])]
    x=compare_experiments(r,prediction_columns=["a","b"],task="regression",metric="rmse",bootstrap=120,permutation=120,seed=2)
    y=compare_experiments(r,prediction_columns=["a","b"],task="regression",metric="rmse",bootstrap=120,permutation=120,seed=2)
    assert x["pairwise"]==y["pairwise"] and x["models"]==y["models"]

 def test_selection_and_ties(self):
    r=[{"fold":i,"model":m,"score":(i+1 if m=="a" else i+1)} for i in range(5) for m in ("a","b")]
    x=bootstrap_selection(r,bootstrap=300,seed=3); assert x["valid_draws"]>=150 and x["n_models"]==2
    with self.assertRaises(ValueError): bootstrap_selection(r[:-1],bootstrap=100)

 def test_csv_loader_and_all_metrics(self):
    with tempfile.TemporaryDirectory() as td:
     p=Path(td)/"x.csv"; p.write_text("id,y_true,a,b\nr0,0,0,1\nr1,1,1,0\n",encoding="utf-8")
     r=load_records(p,fmt="csv"); self.assertIsInstance(r[0]["y_true"],int)
     for metric in ("accuracy","roc_auc","brier","log_loss"):
      out=compare_experiments(r,prediction_columns=["a","b"],metric=metric,bootstrap=20,permutation=20)
      self.assertEqual(len(out["pairwise"]),1)
    rr=[{"id":str(i),"y_true":y,"a":p,"b":q} for i,(y,p,q) in enumerate([(1,1,2),(2,2,2),(3,4,3)])]
    for metric in ("mae","rmse","r2"):
     out=compare_experiments(rr,prediction_columns=["a","b"],task="regression",metric=metric,bootstrap=20,permutation=20)
     self.assertTrue(math.isfinite(out["pairwise"][0]["delta"]))

 def test_three_models_and_bad_inputs(self):
    r=rows()
    for i,x in enumerate(r): x["c"]=x["a"]
    self.assertEqual(len(compare_experiments(r,prediction_columns=["a","b","c"],bootstrap=20,permutation=20)["pairwise"]),3)
    for kw in ({"bootstrap":0},{"bootstrap":-1},{"bootstrap":1.5},{"bootstrap":True},{"permutation":0},{"alpha":float("nan")},{"seed":True}):
     with self.assertRaises(ValueError): compare_experiments(r,prediction_columns=["a","b"],**kw)
    with self.assertRaises(ValueError): compare_experiments([],prediction_columns=["a","b"])
    with self.assertRaises(ValueError): compare_experiments([1],prediction_columns=["a","b"])
    with self.assertRaises(ValueError): compare_experiments(r,prediction_columns=["a","a"])
    const=[{"id":str(i),"y_true":1,"a":i,"b":i} for i in range(3)]
    with self.assertRaises(ValueError): compare_experiments(const,prediction_columns=["a","b"],task="regression",metric="r2")

 def test_selection_oracle_and_bad_folds(self):
    r=[{"fold":f,"model":m,"score":(f+1 if m=="a" else 0)} for f in range(4) for m in ("a","b")]
    out=bootstrap_selection(r,bootstrap=200,seed=7)
    self.assertAlmostEqual(out["point_estimate"],2.5714285714285716)
    with self.assertRaises(ValueError): bootstrap_selection(r+[ {"fold":0,"model":"a","score":1}],bootstrap=200)
    bad=[{"fold":"","model":"a","score":1},{"fold":"","model":"b","score":1}]
    with self.assertRaises(ValueError): bootstrap_selection(bad,bootstrap=200)
    one=[{"fold":0,"model":m,"score":1} for m in ("a","b")]
    with self.assertRaises(ValueError): bootstrap_selection(one,bootstrap=200)

class DiagnosticRegressionTests(unittest.TestCase):
 def test_custom_fold_validation_and_coverage(self):
    r=rows()
    for i,row in enumerate(r): row.update(cv=i % 2, cohort="A" if i < 2 else "B")
    r[3].pop("cohort")
    out=compare_experiments(r,prediction_columns=["a","b"],metric="accuracy",fold_column="cv",group_column="cohort",bootstrap=10,permutation=10)
    self.assertEqual(out["diagnostics"]["fold"], {"column":"cv","missing":0,"unique":2,"counts":[{"value":0,"rows":2},{"value":1,"rows":2}]})
    self.assertEqual(out["diagnostics"]["group"]["missing"],1)
    self.assertEqual(out["diagnostics"]["group"]["counts"],[{"value":"A","rows":2},{"value":"B","rows":1}])
    r[0]["cv"]=-1
    with self.assertRaisesRegex(ValueError,"cv must be"):
     compare_experiments(r,prediction_columns=["a","b"],metric="accuracy",fold_column="cv",bootstrap=10,permutation=10)

"""Saklanan en iyi modeli test kümesinde değerlendirir."""
from trainlib import *
import json, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *
def main():
    ck=torch.load(ROOT/"models/best_model.pth",map_location="cpu",weights_only=False); classes=ck["classes"]; cfg=ck["config"]; device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=M.SARClassifier(len(classes)).to(device); model.load_state_dict(ck["model_state"]); _,_,test=loaders(load_manifest(),classes,cfg,ck.get("preprocessing","raw")); y,p,prob,paths,lat=infer(model,test,device)
    metrics={"accuracy":accuracy_score(y,p),"precision_macro":precision_score(y,p,average="macro",zero_division=0),"recall_macro":recall_score(y,p,average="macro",zero_division=0),"macro_f1":f1_score(y,p,average="macro",zero_division=0),"weighted_f1":f1_score(y,p,average="weighted",zero_division=0),"average_inference_seconds":lat}
    if len(classes)==2:
        metrics["roc_auc"]=roc_auc_score(y,np.asarray(prob)[:,1]); fpr,tpr,_=roc_curve(y,np.asarray(prob)[:,1]); plt.plot(fpr,tpr); plt.savefig(ROOT/"results/figures/roc_curve.png"); plt.close(); prec,rec,_=precision_recall_curve(y,np.asarray(prob)[:,1]); plt.plot(rec,prec); plt.savefig(ROOT/"results/figures/pr_curve.png"); plt.close()
    (ROOT/"results/tables/test_metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8"); pd.DataFrame(classification_report(y,p,target_names=classes,output_dict=True,zero_division=0)).T.to_csv(ROOT/"results/tables/classification_report.csv")
    cm=confusion_matrix(y,p); sns.heatmap(cm,annot=True,fmt="d",xticklabels=classes,yticklabels=classes); plt.tight_layout(); plt.savefig(ROOT/"results/figures/confusion_matrix.png"); plt.close(); wrong=[f"- {path}: gerçek={classes[a]}, tahmin={classes[b]}, güven={max(pr):.4f}" for a,b,pr,path in zip(y,p,prob,paths) if a!=b]; (ROOT/"results/model_evaluation.md").write_text("# Model Değerlendirmesi\n\n```json\n"+json.dumps(metrics,indent=2)+"\n```\n\n## Hatalı örnekler\n"+"\n".join(wrong),encoding="utf-8"); print(metrics)
if __name__=="__main__": main()

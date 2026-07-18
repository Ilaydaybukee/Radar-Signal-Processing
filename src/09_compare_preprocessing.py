"""Ön işleme yöntemlerini aynı split, seed ve mimari ile karşılaştırır."""
from trainlib import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,accuracy_score

def main():
    rows=[]
    for pre in ["raw","log_normalized","median","clahe","lee","sharpened"]:
        model,classes,test,valf1,elapsed,device,_=train_experiment(pre,False); y,p,_,_,lat=infer(model,test,device); rows.append({"preprocessing":pre,"validation_macro_f1":valf1,"test_macro_f1":f1_score(y,p,average="macro",zero_division=0),"accuracy":accuracy_score(y,p),"training_seconds":elapsed,"mean_inference_seconds":lat})
    df=pd.DataFrame(rows); df.to_csv(ROOT/"results/tables/preprocessing_comparison.csv",index=False); df.plot.bar(x="preprocessing",y=["validation_macro_f1","test_macro_f1","accuracy"]); plt.tight_layout(); plt.savefig(ROOT/"results/figures/preprocessing_comparison.png"); (ROOT/"results/preprocessing_report.md").write_text("# Ön İşleme Karşılaştırması\n\n"+df.to_markdown(index=False),encoding="utf-8")
if __name__=="__main__": main()

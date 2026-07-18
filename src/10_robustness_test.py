"""Kontrollü blur, speckle, kontrast ve parlaklık dayanıklılığı."""
from trainlib import *
import torch, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,accuracy_score

def main():
    ck=torch.load(ROOT/"models/best_model.pth",map_location="cpu",weights_only=False); classes=ck["classes"]; device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=M.SARClassifier(len(classes)).to(device); model.load_state_dict(ck["model_state"]); mf=load_manifest(); testf=mf[mf.split=="test"]; rows=[]
    cases=[("blur",str(s),lambda x,s=s: cv2.GaussianBlur(x,(0,0),s)) for s in (.5,1,1.5,2)]+[("speckle",str(s),lambda x,s=s: np.clip(x+x*np.random.default_rng(42).normal(0,s,x.shape),0,1)) for s in (.05,.1,.2)]+[("contrast","0.7",lambda x:(x-.5)*.7+.5),("brightness","0.7",lambda x:x*.7)]
    for kind,level,fn in cases:
        ds=D.SARDataset(testf,classes,ck["config"]["image_size"],False,ck.get("preprocessing","raw"),fn); loader=torch.utils.data.DataLoader(ds,batch_size=ck["config"]["batch_size"]); y,p,_,_,_=infer(model,loader,device); rows.append({"bozulma":kind,"seviye":level,"accuracy":accuracy_score(y,p),"macro_f1":f1_score(y,p,average="macro",zero_division=0)})
    df=pd.DataFrame(rows); df.to_csv(ROOT/"results/tables/robustness_results.csv",index=False)
    for kind,name in [("blur","blur_robustness.png"),("speckle","speckle_robustness.png")]: df[df.bozulma==kind].plot(x="seviye",y="macro_f1",marker="o"); plt.tight_layout(); plt.savefig(ROOT/"results/figures"/name); plt.close()
    (ROOT/"results/robustness_report.md").write_text("# Dayanıklılık Raporu\n\n"+df.to_markdown(index=False),encoding="utf-8")
if __name__=="__main__": main()

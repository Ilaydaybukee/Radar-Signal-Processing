"""Hash gruplarını bölmeden, sınıf oranlarını koruyarak manifest hazırlar."""
from common import *
from sklearn.model_selection import train_test_split

def main():
    p=ROOT/"results/tables/image_metadata.csv"
    if not p.exists(): raise SystemExit("Önce veri denetimini çalıştırın.")
    df=pd.read_csv(p); df=df[df.okunabilir.astype(str).str.lower().eq("true") & df.gecersizlik_nedeni.fillna("").eq("")].copy()
    task,classes,reason=decide_task(df.muhtemel_gemi_turu)
    if not task: write_problem(reason); print(reason); return
    df=df[df.muhtemel_gemi_turu.isin(classes)].drop_duplicates("hash"); counts=df.muhtemel_gemi_turu.value_counts()
    if (counts<3).any(): write_problem("Her sınıfta güvenli 70/15/15 bölme için en az 3 benzersiz görüntü gerekir."); return
    train,temp=train_test_split(df,test_size=.30,random_state=42,stratify=df.muhtemel_gemi_turu)
    val,test=train_test_split(temp,test_size=.50,random_state=42,stratify=temp.muhtemel_gemi_turu)
    out=pd.concat([train.assign(split="train"),val.assign(split="validation"),test.assign(split="test")]); out["label"]=out.muhtemel_gemi_turu; out["task"]=task
    assert out.groupby("hash").split.nunique().max()==1
    out.to_csv(ROOT/"results/tables/split_manifest.csv",index=False); print(out.groupby(["split","label"]).size())
if __name__=="__main__": main()

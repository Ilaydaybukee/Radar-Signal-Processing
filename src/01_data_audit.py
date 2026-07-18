"""Ham TIFF envanteri, bütünlük ve etiket denetimi."""
from common import *
import numpy as np
import pandas as pd
import tifffile
import re

def main():
    rows=[]; invalid=[]
    for p in tiffs():
        base={"dosya_adi":p.name,"dosya_yolu":str(p.resolve()),"uzanti":p.suffix.lower(),"dosya_boyutu":p.stat().st_size,"hash":sha256(p)}
        low=p.stem.lower(); base.update({"muhtemel_gemi_turu":label_from_name(p.name),"muhtemel_sar_bandi":next((b.upper() for b in ("sband","cband","xband","lband") if b in low),None),"muhtemel_polarizasyon":next((x.upper() for x in ("hh","hv","vh","vv") if re.search(rf"(^|[_-]){x}([_-]|$)",low)),None),"muhtemel_veri_kaynagi":next((x for x in ("sentinel","novasar","alos","palsar","nastar") if x in low),None)})
        try:
            a=np.asarray(tifffile.imread(p)); finite=np.nan_to_num(a.astype(float)); shape=a.shape
            channels=shape[-1] if a.ndim==3 and shape[-1]<=4 else (shape[0] if a.ndim==3 and shape[0]<=4 else 1)
            h,w=(shape[-2],shape[-1]) if channels==1 else (shape[0],shape[1])
            mn,mx=float(np.min(finite)),float(np.max(finite)); std=float(np.std(finite))
            reason=[]
            if std==0: reason.append("sıfır varyans")
            if mx==0: reason.append("tamamen siyah")
            if np.issubdtype(a.dtype,np.integer) and mn==mx==np.iinfo(a.dtype).max: reason.append("tamamen beyaz")
            base.update({"genislik":w,"yukseklik":h,"kanal_sayisi":channels,"numpy_veri_tipi":str(a.dtype),"minimum_piksel":mn,"maksimum_piksel":mx,"ortalama":float(np.mean(finite)),"standart_sapma":std,"okunabilir":True,"gecersizlik_nedeni":"; ".join(reason)})
        except Exception as e: base.update({"okunabilir":False,"gecersizlik_nedeni":f"bozuk/okunamıyor: {e}"})
        rows.append(base)
        if base.get("gecersizlik_nedeni"): invalid.append(base)
    out=ROOT/"results/tables"; out.mkdir(parents=True,exist_ok=True); columns=["dosya_adi","dosya_yolu","uzanti","genislik","yukseklik","kanal_sayisi","numpy_veri_tipi","minimum_piksel","maksimum_piksel","ortalama","standart_sapma","dosya_boyutu","okunabilir","hash","muhtemel_sar_bandi","muhtemel_gemi_turu","muhtemel_polarizasyon","muhtemel_veri_kaynagi","gecersizlik_nedeni"]; df=pd.DataFrame(rows,columns=columns); df.to_csv(out/"image_metadata.csv",index=False)
    dup=df[df.duplicated("hash",keep=False)] if len(df) else df.copy(); dup.to_csv(out/"duplicate_images.csv",index=False); pd.DataFrame(invalid).to_csv(out/"invalid_images.csv",index=False)
    task,classes,reason=decide_task(df.get("muhtemel_gemi_turu",[])); counts=df.get("muhtemel_gemi_turu",pd.Series(dtype=str)).value_counts().to_dict()
    summary=f"# Veri Denetimi Özeti\n\n- Toplam: {len(df)}\n- Sağlam: {len(df)-len(invalid)}\n- Geçersiz: {len(invalid)}\n- Tekrar kayıtları: {len(dup)}\n- Sınıflar: {counts}\n- Görev: {task or 'eğitim yapılamaz'}\n- Karar: {reason}\n"
    (ROOT/"results/audit_summary.md").write_text(summary,encoding="utf-8")
    if not task: write_problem(reason)
    print(summary)
if __name__=="__main__": main()

"""Ham değerleri değiştirmeden veri seti grafiklerini üretir."""
from common import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_bar(series,title,name):
    plt.figure(figsize=(8,4)); series.fillna("Bilinmiyor").value_counts().plot.bar(); plt.title(title); plt.tight_layout(); plt.savefig(ROOT/"results/figures"/name,dpi=150); plt.close()
def main():
    seed_everything(); p=ROOT/"results/tables/image_metadata.csv"
    if not p.exists(): raise SystemExit("Önce veri denetimini çalıştırın.")
    df=pd.read_csv(p); good=df[df.okunabilir.astype(str).str.lower().eq("true")]
    for col,title,name in [("muhtemel_gemi_turu","Sınıf dağılımı","class_distribution.png"),("muhtemel_sar_bandi","Bant dağılımı","band_distribution.png"),("numpy_veri_tipi","Veri tipi dağılımı","dtype_distribution.png")]: save_bar(good[col],title,name)
    plt.figure(); plt.scatter(good.genislik,good.yukseklik,alpha=.6); plt.xlabel("Genişlik"); plt.ylabel("Yükseklik"); plt.tight_layout(); plt.savefig(ROOT/"results/figures/image_sizes.png"); plt.close()
    samples=good.groupby("muhtemel_gemi_turu",dropna=False).sample(n=1,random_state=42) if len(good) else good
    if len(samples):
        fig,axs=plt.subplots(1,len(samples),figsize=(4*len(samples),4)); axs=np.atleast_1d(axs)
        for ax,(_,r) in zip(axs,samples.iterrows()): ax.imshow(normalize(read_gray(r.dosya_yolu)),cmap="gray"); ax.set_title(str(r.muhtemel_gemi_turu)); ax.axis("off")
        fig.tight_layout(); fig.savefig(ROOT/"results/figures/class_samples.png",dpi=150); plt.close(fig)
        plt.figure();
        for _,r in samples.iterrows(): plt.hist(normalize(read_gray(r.dosya_yolu)).ravel(),bins=64,alpha=.35,label=str(r.muhtemel_gemi_turu))
        plt.legend(); plt.title("Gösterim-normalize piksel histogramları"); plt.tight_layout(); plt.savefig(ROOT/"results/figures/pixel_histograms.png"); plt.close()
    print(f"{len(good)} okunabilir görüntü için ön izleme tamamlandı.")
if __name__=="__main__": import numpy as np, pandas as pd; main()

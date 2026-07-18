"""Her filtreyi ham görüntüden bağımsız üretir; ham dosyaya asla yazmaz."""
from common import *
from scipy.ndimage import median_filter, gaussian_filter, uniform_filter
from skimage import exposure, restoration, filters
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def lee(x,size=5):
    mean=uniform_filter(x,size); mean2=uniform_filter(x*x,size); var=np.maximum(mean2-mean*mean,0); noise=np.mean(var); return mean+(var/(var+noise+1e-8))*(x-mean)
def methods(x):
    return {"percentile_normalized":normalize(x),"log_normalized":np.log1p(normalize(x)*255)/np.log(256),"median":median_filter(x,size=3),"gaussian":gaussian_filter(x,sigma=1),"clahe":exposure.equalize_adapthist(normalize(x),clip_limit=.03),"lee":lee(x.astype(np.float32)),"sharpened":filters.unsharp_mask(x.astype(np.float32),radius=1,amount=1,preserve_range=True),"bilateral":restoration.denoise_bilateral(normalize(x),sigma_color=.05,sigma_spatial=3,channel_axis=None)}
def cast_like(y,a):
    y=np.nan_to_num(y); 
    if np.issubdtype(a.dtype,np.integer): y=np.clip(y,np.iinfo(a.dtype).min,np.iinfo(a.dtype).max).astype(a.dtype)
    return y.astype(a.dtype,copy=False)
def main():
    rows=[]; paths=tiffs()
    for i,p in enumerate(paths):
        a=read_gray(p)
        for name,y in methods(a).items():
            out=ROOT/"data/processed"/name/(p.stem+".tif"); out.parent.mkdir(parents=True,exist_ok=True); tifffile.imwrite(out,cast_like(y,a))
            rows.append({"dosya":str(p),"yontem":name,"once_min":float(np.min(a)),"once_max":float(np.max(a)),"once_ortalama":float(np.mean(a)),"once_std":float(np.std(a)),"once_varyans":float(np.var(a)),"sonra_min":float(np.min(y)),"sonra_max":float(np.max(y)),"sonra_ortalama":float(np.mean(y)),"sonra_std":float(np.std(y)),"sonra_varyans":float(np.var(y)),"nan_sayisi":int(np.isnan(y).sum()),"inf_sayisi":int(np.isinf(y).sum())})
        if i<10:
            fig,axs=plt.subplots(3,3,figsize=(10,10)); axs.flat[0].imshow(normalize(a),cmap="gray"); axs.flat[0].set_title("Ham")
            for ax,(name,y) in zip(axs.flat[1:],methods(a).items()): ax.imshow(normalize(y),cmap="gray"); ax.set_title(name)
            for ax in axs.flat: ax.axis("off")
            fig.tight_layout(); fig.savefig(ROOT/"results/figures"/f"processing_{i:02d}.png"); plt.close(fig)
    pd.DataFrame(rows).to_csv(ROOT/"results/tables/processing_statistics.csv",index=False); print(f"{len(paths)} görüntü işlendi.")
if __name__=="__main__": import pandas as pd, tifffile; main()

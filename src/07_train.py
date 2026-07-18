"""Yapılandırmalı CUDA/AMP eğitiminin komut satırı girişi."""
from trainlib import *
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
if __name__=="__main__":
    model,classes,test,best,elapsed,device,h=train_experiment(config().get("preprocessing","raw"),True); df=pd.DataFrame(h); df.plot(x="epoch",y=["train_loss"]); plt.tight_layout(); plt.savefig(ROOT/"results/figures/loss_curve.png"); plt.close(); df.plot(x="epoch",y=["validation_accuracy","validation_macro_f1"]); plt.tight_layout(); plt.savefig(ROOT/"results/figures/accuracy_curve.png"); print(f"En iyi macro F1={best:.4f}; aygıt={device}; süre={elapsed:.1f}s")

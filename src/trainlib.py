"""Eğitim ve değerlendirmede paylaşılan PyTorch yordamları."""
from common import *
import importlib.util, time, torch
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader

def module(file,name): spec=importlib.util.spec_from_file_location(name,ROOT/"src"/file); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
D=module("05_dataset.py","sar_dataset"); M=module("06_model.py","sar_model")
def loaders(manifest,classes,cfg,pre="raw"):
    mk=lambda split,aug: DataLoader(D.SARDataset(manifest[manifest.split==split],classes,cfg["image_size"],aug,pre),batch_size=cfg["batch_size"],shuffle=aug,num_workers=cfg["num_workers"],pin_memory=True)
    return mk("train",True),mk("validation",False),mk("test",False)
def infer(model,loader,device):
    model.eval(); ys=[]; ps=[]; probs=[]; paths=[]; start=time.perf_counter()
    with torch.no_grad():
        for x,y,p in loader:
            z=torch.softmax(model(x.to(device)),1).cpu(); ys+=y.tolist(); ps+=z.argmax(1).tolist(); probs+=z.tolist(); paths+=list(p)
    return ys,ps,probs,paths,(time.perf_counter()-start)/max(len(ys),1)
def train_experiment(pre="raw",save=True):
    cfg=config(); seed_everything(cfg["seed"]); manifest=load_manifest(); classes=sorted(manifest.label.unique());
    if cfg.get("require_cuda",True) and not torch.cuda.is_available(): raise RuntimeError("CUDA bulunamadı. RTX 4070/CUDA PyTorch kurulumu doğrulanmadan eğitim başlatılmadı.")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); train,val,test=loaders(manifest,classes,cfg,pre); model=M.SARClassifier(len(classes)).to(device)
    loss_fn=torch.nn.CrossEntropyLoss(weight=D.class_weights(manifest[manifest.split=="train"],classes).to(device)); opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"]); sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",patience=3); scaler=torch.amp.GradScaler("cuda",enabled=cfg["use_amp"] and device.type=="cuda")
    best=-1.; stale=0; history=[]; started=time.perf_counter()
    for epoch in range(1,cfg["epochs"]+1):
        model.train(); total=0
        for x,y,_ in train:
            x,y=x.to(device),y.to(device); opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type,enabled=scaler.is_enabled()): loss=loss_fn(model(x),y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); total+=loss.item()*len(y)
        vy,vp,_,_,_=infer(model,val,device); f1=f1_score(vy,vp,average="macro",zero_division=0); acc=accuracy_score(vy,vp); sched.step(f1); history.append({"epoch":epoch,"train_loss":total/len(train.dataset),"validation_accuracy":acc,"validation_macro_f1":f1}); print(f"Epoch {epoch}: loss={history[-1]['train_loss']:.4f}, val_accuracy={acc:.4f}, val_macro_f1={f1:.4f}")
        if f1>best: best=f1; stale=0; best_state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        else: stale+=1
        if stale>=cfg["patience"]: break
    model.load_state_dict(best_state); elapsed=time.perf_counter()-started
    if save: torch.save({"model_state":best_state,"classes":classes,"config":cfg,"preprocessing":pre,"task":manifest.task.iloc[0]},ROOT/"models/best_model.pth"); pd.DataFrame(history).to_csv(ROOT/"results/tables/training_history.csv",index=False)
    return model,classes,test,best,elapsed,device,history

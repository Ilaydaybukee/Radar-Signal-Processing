"""Tek kanallı TIFF PyTorch veri kümesi."""
from common import *
import cv2, torch
from torch.utils.data import Dataset

def letterbox(x,size=256):
    h,w=x.shape; scale=min(size/h,size/w); nh,nw=max(1,round(h*scale)),max(1,round(w*scale)); y=cv2.resize(x,(nw,nh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR); out=np.zeros((size,size),np.float32); top=(size-nh)//2; left=(size-nw)//2; out[top:top+nh,left:left+nw]=y; return out
class SARDataset(Dataset):
    def __init__(self,frame,classes,image_size=256,augment=False,preprocessing="raw",corruption=None): self.frame=frame.reset_index(drop=True); self.classes=list(classes); self.image_size=image_size; self.augment=augment; self.preprocessing=preprocessing; self.corruption=corruption
    def __len__(self): return len(self.frame)
    def __getitem__(self,i):
        r=self.frame.iloc[i]; path=Path(r.dosya_yolu)
        if self.preprocessing!="raw":
            candidate=ROOT/"data/processed"/self.preprocessing/(path.stem+".tif")
            if candidate.exists(): path=candidate
        x=letterbox(normalize(read_gray(path)),self.image_size)
        if self.augment:
            if np.random.rand()<.5: x=np.fliplr(x).copy()
            if np.random.rand()<.5: x=np.flipud(x).copy()
            angle=np.random.uniform(-10,10); m=cv2.getRotationMatrix2D((self.image_size/2,)*2,angle,1); x=cv2.warpAffine(x,m,(self.image_size,)*2)
            if np.random.rand()<.25: x=cv2.GaussianBlur(x,(3,3),0.5)
            if np.random.rand()<.25: x=np.clip(x+x*np.random.normal(0,.08,x.shape),0,1)
        if self.corruption: x=self.corruption(x)
        return torch.from_numpy(np.ascontiguousarray(x,dtype=np.float32)[None]), self.classes.index(r.label), str(r.dosya_yolu)
def class_weights(frame,classes):
    counts=frame.label.value_counts(); return torch.tensor([len(frame)/(len(classes)*counts[c]) for c in classes],dtype=torch.float32)

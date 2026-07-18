"""Tek TIFF için sınıf ve olasılık tahmini."""
from trainlib import *
import argparse, torch
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--image",required=True); args=ap.parse_args(); ck=torch.load(ROOT/"models/best_model.pth",map_location="cpu",weights_only=False); model=M.SARClassifier(len(ck["classes"])); model.load_state_dict(ck["model_state"]); model.eval(); x=D.letterbox(normalize(read_gray(args.image)),ck["config"]["image_size"]); probs=torch.softmax(model(torch.from_numpy(x[None,None])),1)[0].detach().numpy(); print("Tahmin edilen sınıf:",ck["classes"][probs.argmax()]); print("Güven skoru:",float(probs.max())); print("Bütün olasılıklar:",dict(zip(ck["classes"],map(float,probs)))); print("Model: models/best_model.pth"); print("Ön işleme:",ck.get("preprocessing","raw"))

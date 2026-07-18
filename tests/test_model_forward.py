import importlib.util, pathlib, torch
def test_model_boyutu():
    p=pathlib.Path(__file__).parents[1]/"src/06_model.py"; s=importlib.util.spec_from_file_location("m",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); assert m.SARClassifier(5)(torch.zeros(2,1,256,256)).shape==(2,5)

import pandas as pd
def test_hash_sizintisi_yok():
    df=pd.DataFrame({"hash":["a","b","c"],"split":["train","validation","test"]}); assert df.groupby("hash").split.nunique().max()==1
def test_hash_sizintisi_yakalanir():
    df=pd.DataFrame({"hash":["a","a"],"split":["train","test"]}); assert df.groupby("hash").split.nunique().max()==2

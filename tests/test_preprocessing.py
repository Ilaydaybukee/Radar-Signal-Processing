import numpy as np
from common import normalize
def test_normalizasyon_sinirli_ve_sonlu():
    a=np.array([[0,np.nan],[100,np.inf]],dtype=np.float32); y=normalize(a); assert np.isfinite(y).all() and 0<=y.min()<=y.max()<=1
def test_sifir_varyans_nan_uretmez(): assert np.isfinite(normalize(np.ones((4,4),dtype=np.uint16))).all()

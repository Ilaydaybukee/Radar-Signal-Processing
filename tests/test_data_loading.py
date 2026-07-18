import numpy as np, tifffile
from common import read_gray
def test_tiff_okuma(tmp_path):
    p=tmp_path/"ornek.tif"; tifffile.imwrite(p,np.arange(100,dtype=np.uint16).reshape(10,10)); a=read_gray(p); assert a.shape==(10,10) and a.dtype==np.uint16

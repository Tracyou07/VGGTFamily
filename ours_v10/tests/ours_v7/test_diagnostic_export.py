import io
import unittest
import numpy as np
from experiments.ours_v7 import diagnostic_export as e
class ExportTest(unittest.TestCase):
    def test_lossless_noncontiguous_alias_unicode_and_empty(self):
        a=np.arange(24,dtype=np.float32).reshape(4,6)[:,::2]
        values=dict(depth=a,confidence=a,frame_ids=np.array(['000000','000001']),empty=np.zeros((0,3)))
        stream=io.BytesIO()
        e.save_npz_fast(stream,**values)
        stream.seek(0)
        with np.load(stream,allow_pickle=False) as z:
            self.assertEqual(set(z.files),set(values))
            for k,v in values.items():np.testing.assert_array_equal(z[k],v)
        np.testing.assert_array_equal(a,np.arange(24,dtype=np.float32).reshape(4,6)[:,::2])
if __name__=='__main__':unittest.main()

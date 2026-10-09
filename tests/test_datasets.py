import unittest
import numpy as np
from datasets import four_cells,noise_pair,continuous_shapes


class GeneratedDataTests(unittest.TestCase):
    def test_four_cell_gains(self):
        x,y=four_cells(1)
        def gain(feature,mask):
            left=y[mask & (x[:,feature]<0)]; right=y[mask & (x[:,feature]>0)]
            return len(left)*len(right)/(len(left)+len(right))*(left.mean()-right.mean())**2
        self.assertEqual(x.shape,(512,2))
        self.assertEqual(gain(0,np.ones(512,dtype=bool)),512)
        self.assertEqual(gain(1,np.ones(512,dtype=bool)),0)
        self.assertEqual(gain(1,x[:,0]<0),256)

    def test_matched_noise(self):
        _,y=four_cells(1)
        iid,odd=noise_pair(y)
        np.testing.assert_array_equal(odd[:256],-odd[256:])
        np.testing.assert_allclose(np.mean(iid**2),np.mean(odd**2),rtol=2e-15)

    def test_continuous_mirrors_and_reproducibility(self):
        x,y,ids=continuous_shapes()
        np.testing.assert_array_equal(y[::3],-y[2::3])
        np.testing.assert_array_equal(y[1::3],0)
        np.testing.assert_array_equal(x[::3,1:],x[2::3,1:])
        np.testing.assert_array_equal(x,continuous_shapes()[0])
        self.assertFalse(np.array_equal(x,continuous_shapes(seed=20261010)[0]))


if __name__=='__main__':
    unittest.main()

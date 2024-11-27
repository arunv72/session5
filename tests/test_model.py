import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import torch
from mnist_dnn import MNIST_DNN

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = MNIST_DNN()
        
    def test_parameter_count(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        self.assertTrue(25000 <= total_params <= 26000, 
                       f"Parameter count {total_params} is not in range [25000, 26000]")
        
    def test_input_shape(self):
        batch_size = 1
        input_tensor = torch.randn(batch_size, 1, 28, 28)
        try:
            output = self.model(input_tensor)
        except:
            self.fail("Model failed to process 28x28 input")
            
    def test_output_shape(self):
        batch_size = 1
        input_tensor = torch.randn(batch_size, 1, 28, 28)
        output = self.model(input_tensor)
        self.assertEqual(output.shape, (batch_size, 10), 
                        f"Output shape {output.shape} is not correct")

if __name__ == '__main__':
    unittest.main() 
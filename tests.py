import torch, unittest
from unets import thin_setup, Unet, ThinUnetUpBlock, ThinUnetDownBlock

class BaseTests(unittest.TestCase):
    def test_inequal_output_asymmetric(self):
        unet = Unet(
            in_features=3,
            down=[16, 32, 64], up=[40, 4]
        )
        input = torch.zeros(2, 3, 104, 104)
        output = unet(input)
        self.assertEqual(torch.Size([2, 4, 24, 24]), output.size())

    def test_inequal_output_symmetric(self):
        unet = Unet(
            down=[16, 32, 64], up=[40, 1]
        )
        input = torch.zeros(2, 1, 104, 104)
        output = unet(input)
        self.assertEqual(torch.Size([2, 1, 24, 24]), output.size())

class ThinTests(unittest.TestCase):
    def test_inequal_output_asymmetric(self):
        unet = Unet(
            in_features=3,
            down=[16, 32, 64],
            up=[40, 4],
            setup=thin_setup
        )
        input = torch.zeros(2, 3, 104, 104)
        output = unet(input)
        self.assertEqual(torch.Size([2, 4, 64, 64]), output.size())

    def test_inequal_output_symmetric(self):
        unet = Unet(
            down=[16, 32, 64],
            up=[40, 1],
            setup=thin_setup
        )
        input = torch.zeros(2, 1, 104, 104)
        output = unet(input)
        self.assertEqual(torch.Size([2, 1, 64, 64]), output.size())

unittest.main()

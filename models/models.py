import torch
NUM_DIGITS = 10


class FizzBuzzModelV3(torch.nn.Module):
    def __init__(self, num_digits):
        super(FizzBuzzModelV3, self).__init__()
        self.l1 = torch.nn.Linear(num_digits, 256)
        self.relu = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(256, 64)
        self.l3 = torch.nn.Linear(64, 4)

    def init_weights(self):
        self.l1.weight.data.normal_(0.0, 0.01)
        self.l2.weight.data.normal_(0.0, 0.01)
        self.l3.weight.data.normal_(0.0, 0.01)
        self.l1.bias.data.fill_(0)
        self.l2.bias.data.fill_(0)
        self.l3.bias.data.fill_(0)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out
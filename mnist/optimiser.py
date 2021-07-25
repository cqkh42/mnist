class SGD:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def step(self):
        for param in self.params:
            param.data -= param.grad.data * self.lr

    def zero_grad(self):
        for param in self.params:
            param.grad = None

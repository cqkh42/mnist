class Learner:
    def __init__(self, lr, epochs, optimiser, model, loss):
        self.epochs = epochs
        self.lr = lr
        self.optimiser_class = optimiser
        self.model = model
        self.optimiser = None
        if model:
            self.load_optimiser()
        self.loss_func = loss

    def load_optimiser(self):
        self.optimiser = self.optimiser_class(
            self.model.parameters(), self.lr
        )

    def _calc_gradient(self, X, y):
        epoch_loss = self.loss(X, y)
        epoch_loss.backward()

    def loss(self, X, y_true):
        preds = self.model(X)
        return self.loss_func(preds, y_true)

    def fit_batch(self, X, y):
        self._calc_gradient(X, y)
        self.optimiser.step()
        self.optimiser.zero_grad()

    def _fit(self, dl):
        for _ in range(self.epochs):
            for X, y in dl:
                self.fit_batch(X, y)

    def fit(self, dl):
        self._fit(dl)

    def predict(self, X):
        return self.model(X)
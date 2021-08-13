class Learner:
    def __init__(self, lr, epochs, optimiser, model, loss):
        self.epochs = epochs
        self.predict = model
        self.optimiser = optimiser(model.parameters(), lr)
        self.loss = loss

    def _calc_gradient(self, X, y):
        preds = self.predict(X)
        epoch_loss = self.loss(preds, y)
        epoch_loss.backward()

    def fit_batch(self, X, y):
        self._calc_gradient(X, y)
        self.optimiser.step()
        self.optimiser.zero_grad()

    def fit(self, dl):
        for _ in range(self.epochs):
            for X, y in dl:
                self.fit_batch(X, y)

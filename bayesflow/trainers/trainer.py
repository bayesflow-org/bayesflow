from bayesflow._backend import jit, value_and_grad


class Trainer:
    def __init__(self, loss_fn, optimizer):
        self.optimizer = optimizer
        self.loss_fn = jit(loss_fn)
        self.grad_fn = jit(value_and_grad(loss_fn, argnums=0, has_aux=True))

    def train_step(self, state, batch):
        model_state, optimizer_state = state
        params, frozen = model_state
        (loss, (metrics, frozen)), grads = self.grad_fn(params, frozen, batch)
        params, optimizer_state = self.optimizer.stateless_apply(optimizer_state, grads, params)

        state = (params, frozen), optimizer_state
        return state, metrics

    def fit(self, state, data, epochs=1):
        for epoch in range(epochs):
            for batch in data:
                state, metrics = self.train_step(state, batch)

        return state

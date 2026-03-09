import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras

from bayesflow.trainers import Trainer


class MyModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.linear = keras.layers.Dense(1)

    def build(self, input_shape):
        self.linear.build(input_shape)

    def call(self, x):
        return self.linear(x)


model = MyModel()
model.build((None, 2))


def loss_fn(params, frozen, batch):
    x, y = batch
    y_pred, frozen = model.stateless_call(params, frozen, x)

    mse = keras.losses.mean_squared_error(y, y_pred)
    loss = keras.ops.mean(mse, axis=0)
    metrics = {}
    return loss, (metrics, frozen)


optimizer = keras.optimizers.Adam(learning_rate=1e-3)
optimizer.build(model.trainable_variables)

trainer = Trainer(loss_fn, optimizer)

data = [(keras.ops.zeros((32, 2)), keras.ops.zeros((32, 1)))]

model_state = model.trainable_variables, model.non_trainable_variables
optimizer_state = optimizer.variables

state = model_state, optimizer_state

state = keras.tree.map_structure(lambda v: v.value, state)
print(state)

state = trainer.fit(state, data)

print(state)

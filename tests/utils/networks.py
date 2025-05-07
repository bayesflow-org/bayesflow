from bayesflow.networks import SummaryNetwork
import keras


class MeanStdSummaryNetwork(SummaryNetwork):
    def call(self, x):
        summary_outputs = keras.ops.stack([keras.ops.mean(x, axis=-1), keras.ops.std(x, axis=-1)], axis=-1)
        return summary_outputs

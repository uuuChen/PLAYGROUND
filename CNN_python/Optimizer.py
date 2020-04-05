import numpy as np


class Optimizer:

    def __init__(self):
        pass

    @staticmethod
    def learning_rate_exponential_decay(learning_rate, global_step, decay_steps, decay_rate):
        decayed_learning_rate = learning_rate * np.power(decay_rate, global_step // decay_steps)
        return decayed_learning_rate


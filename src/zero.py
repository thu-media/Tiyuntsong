import sabre
import math
import numpy as np

class Zero(sabre.Abr):

    def __init__(self, config = None):
        pass

    # def create_actor_network(self):

    def get_quality_delay(self, segment_index):
        quality = 1
        delay = 0
        return (quality, delay)

    def update_gradients(self, reward):
        pass
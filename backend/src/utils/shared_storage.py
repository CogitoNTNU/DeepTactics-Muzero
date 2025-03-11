from src.config import Config
from src.networks.network import Network 

class SharedStorage(object):
    def __init__(self):
        self._networks = {}

    def latest_network(self) -> Network:
        if self._networks:
            return self._networks[max(self._networks.keys())]
        else:
            return Network(Config())

    def save_network(self, step: int, network: Network):
        self._networks[step] = network
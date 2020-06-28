from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def build(self):
        print("Model constructed")
        pass

    @abstractmethod
    def compile(self):
        print("Model compiled")
        pass

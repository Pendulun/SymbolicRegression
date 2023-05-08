from abc import ABC, abstractmethod
class IndividualGenerator(ABC):
    @abstractmethod
    def generate(self, *args):
        pass

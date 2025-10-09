from collections.abc import MutableSequence
from cellular.NeighborsTopology import NeighborsTopology
from cellular.TournamentTopology import TournamentTopology
from cellular.factory.NeighborsTopologyFactory import NeighborsTopologyFactory
from typing import TypeVar
import random

T = TypeVar('T')


class TournamentTopologyFactory(NeighborsTopologyFactory):
    def __init__(self,
                 rand: random.Random,
                 pressure: int = 3,
                 ) -> None:
        super().__init__()
        if pressure < 1:
            raise ValueError(f'Pressure must be at least 1, found {pressure} instead.')
        self.__pressure: int = pressure
        self.__rand: random.Random = rand

    def create(self, collection: MutableSequence[T], clone: bool = False) -> NeighborsTopology:
        return TournamentTopology(collection=collection, clone=clone, pressure=self.__pressure, rand=self.__rand)

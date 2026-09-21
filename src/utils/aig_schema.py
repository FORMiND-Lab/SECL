from enum import IntEnum


class GateType(IntEnum):
    PI = 0
    AND = 1
    NOT = 2
    CONST = 3


GATE_TO_INDEX = {gate.name: int(gate) for gate in GateType}
NUM_GATE_TYPES = len(GateType)

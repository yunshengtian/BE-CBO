from .beam_design import WeldedBeamDesign
from .lsq import LSQ, LSQShift
from .simionescu import Simionescu, SimonescuShift
from .speed_reducer import SpeedReducerDesign
from .tension_compression_string import TensionCompressionString
from .townsend import Townsend, TownsendShift
from .vessel_design import VesselDesign
from .bar_truss import ThreeBarTrussDesign
from .rolling_element_bearing import RollingElementBearing
from .gas_transmission_compressor import GasTransmisionCompressor
from .planetary_gear_train import PlanetaryGearTrain
from .cantilever_beam_design import CantileverBeamDesign


test_functions = {

    # synthetic

    # 2D
    'lsq': LSQ,
    'lsq-shift': LSQShift,
    'sim': Simionescu,
    'sim-shift': SimonescuShift,
    'tow': Townsend,
    'tow-shift': TownsendShift,

    # real-world

    # 2D
    '3bar': ThreeBarTrussDesign,

    # 3D
    'ten': TensionCompressionString,

    # 4D
    'beam': WeldedBeamDesign,
    'ves': VesselDesign,
    'gas': GasTransmisionCompressor,

    # 7D
    'spe': SpeedReducerDesign,

    # 9D
    'pla': PlanetaryGearTrain,

    # 10D
    'rol': RollingElementBearing,

    # 30D
    'cbeam': CantileverBeamDesign,
}

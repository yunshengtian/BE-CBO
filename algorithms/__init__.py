from .random import *
from .cei import *
from .scbo import *
from .becbo import *


algorithms = {
    'random-sobol': RandomSobol,
    'cei': CEI,
    'scbo-t-re': SCBO_T_Restart,
    'be-cbo': BECBO,
}

import math
from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    best_constraint_values: Tensor = torch.ones(2,)*torch.inf
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_tr_length(state):
    # Update the length of the trust region according to 
    # success and failure counters 
    # (Just as in original TuRBO paper)
    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min: # Restart when trust region becomes too small 
        state.restart_triggered = True
    
    return state


def update_state(state, Y_next, C_next): 
    ''' Method used to update the TuRBO state after each
        step of optimization. 
        
        Success and failure counters are updated accoding to 
        the objective values (Y_next) and constraint values (C_next) 
        of the batch of candidate points evaluated on the optimization step. 
        
        As in the original TuRBO paper, a success is counted whenver 
        any one of the new candidate points imporves upon the incumbent 
        best point. The key difference for SCBO is that we only compare points 
        by their objective values when both points are valid (meet all constraints). 
        If exactly one of the two points beinc compared voliates a constraint, the 
        other valid point is automatically considered to be better. If both points 
        violate some constraints, we compare them inated by their constraint values. 
        The better point in this case is the one with minimum total constraint violation 
        (the minimum sum over constraint values)'''

    # Determine which candidates meet the constraints (are valid)
    bool_tensor = C_next >= 0 
    bool_tensor = torch.all(bool_tensor, dim=-1)
    Valid_Y_next = Y_next[bool_tensor] 
    Valid_C_next = C_next[bool_tensor]
    if Valid_Y_next.numel() == 0: # if none of the candidates are valid
        # pick the point with max feasibility
        sum_feasible = C_next.sum(dim=-1)
        max_feasible = sum_feasible.max()
        # if the max feasible candidate is larger than the feasibility of the incumbent
        if max_feasible > state.best_constraint_values.sum():
            # count a success and update the current best point and constraint values
            state.success_counter += 1
            state.failure_counter = 0
            # new best is min violator
            state.best_value = Y_next[sum_feasible.argmax()].item()
            state.best_constraint_values = C_next[sum_feasible.argmax()]
        else:
            # otherwise, count a failure
            state.success_counter = 0
            state.failure_counter += 1
    else: # if at least one valid candidate was suggested, 
          # throw out all invalid candidates 
          # (a valid candidate is always better than an invalid one)

        # Case 1: if best valid candidate found has a higher obj value that incumbent best
            # count a success, the obj valuse has been improved
        # improved_obj = max(Valid_Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value)
        improved_obj = max(Valid_Y_next) > state.best_value
        # Case 2: if incumbent best violates constraints
            # count a success, we now have suggested a point which is valid and therfore better
        obtained_validity = torch.any(state.best_constraint_values < 0)
        if improved_obj or obtained_validity: # If Case 1 or Case 2
            # count a success and update the best value and constraint values 
            state.success_counter += 1
            state.failure_counter = 0
            state.best_value = max(Valid_Y_next).item()
            state.best_constraint_values = Valid_C_next[Valid_Y_next.argmax()]
        else:
            # otherwise, count a fialure
            state.success_counter = 0
            state.failure_counter += 1

    # Finally, update the length of the trust region according to the
    # updated success and failure counts
    state = update_tr_length(state) 

    return state   

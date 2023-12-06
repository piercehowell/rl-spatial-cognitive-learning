import inspect
import itertools as itt
import warnings
from functools import partial
from typing import List, Optional, Set, Tuple, Type

import more_itertools as mitt
import numpy as np
import numpy.random as rnd
from typing_extensions import Protocol  # python3.7 compatibility

from gym_gridverse.agent import Agent
from gym_gridverse.design import (
    draw_area,
    draw_line_horizontal,
    draw_line_vertical,
    draw_room_grid,
    draw_wall_boundary,
)
from gym_gridverse.geometry import Orientation, Position, Shape, Area
from gym_gridverse.grid import Grid
from gym_gridverse.grid_object import (
    Beacon,
    Color,
    Door,
    Exit,
    Floor,
    GridObject,
    Key,
    MovingObstacle,
    Telepod,
    Wall,
)
from gym_gridverse.rng import choice, choices, get_gv_rng_if_none, shuffle
from gym_gridverse.state import State
from gym_gridverse.utils.custom import import_if_custom
from gym_gridverse.utils.functions import checkraise_kwargs, select_kwargs
from gym_gridverse.utils.protocols import get_keyword_parameter
from gym_gridverse.utils.registry import FunctionRegistry

from typing import Optional

import numpy.random as rnd

from gym_gridverse.agent import Agent
from gym_gridverse.envs.reset_functions import reset_function_registry
from gym_gridverse.geometry import Orientation, Position
from gym_gridverse.grid import Grid
from gym_gridverse.grid_object import Exit, Floor, Wall
from gym_gridverse.state import State
# TODO: Clean up the imports


# class ResetFunction(Protocol):
#     """Signature that all reset functions must follow"""

#     def __call__(self, *, rng: Optional[rnd.Generator] = None) -> State:
#         ...


# class ResetFunctionRegistry(FunctionRegistry):
#     def get_protocol_parameters(
#         self, signature: inspect.Signature
#     ) -> List[inspect.Parameter]:
#         rng = get_keyword_parameter(signature, 'rng')
#         return [rng]

#     def check_signature(self, function: ResetFunction):
#         signature = inspect.signature(function)
#         (rng,) = self.get_protocol_parameters(signature)

#         # checks `rng` is keyword
#         if rng.kind not in [
#             inspect.Parameter.POSITIONAL_OR_KEYWORD,
#             inspect.Parameter.KEYWORD_ONLY,
#         ]:
#             raise TypeError(
#                 f'The `rng` argument ({rng.name}) '
#                 f'of a registered reward function ({function}) '
#                 'should be allowed to be a keyword argument.'
#             )

#         # checks if annotations, if given, are consistent
#         if rng.annotation not in [
#             inspect.Parameter.empty,
#             Optional[rnd.Generator],
#         ]:
#             warnings.warn(
#                 f'The `rng` argument ({rng.name}) '
#                 f'of a registered reward function ({function}) '
#                 f'has an annotation ({rng.annotation}) '
#                 'which is not `Optional[rnd.Generator]`.'
#             )

#         if signature.return_annotation not in [inspect.Parameter.empty, State]:
#             warnings.warn(
#                 f'The return type of a registered reset function ({function}) '
#                 f'has an annotation ({signature.return_annotation}) '
#                 'which is not `State`.'
#             )


# reset_function_registry = ResetFunctionRegistry()
# """Reset function registry"""


def build_grid_world():
    """
    Builds our custome grid-world
    """
    grid = Grid.from_shape((9, 9), factory=Floor)
    # assigns Wall to the border
    draw_wall_boundary(grid)
    draw_area(grid, Area((2, 4), (5,5)), Wall,fill=True)
    draw_area(grid, Area((6, 7),(3, 3)), Wall,fill=True)
    draw_area(grid, Area((4, 4),(2, 7)), Wall,fill=True)
    grid[3,3]=Wall()
    grid[6,7]=Wall()
    grid[4,7]=Floor()
    return(grid)


@reset_function_registry.register
def landmark_start_and_goal(*, rng: Optional[rnd.Generator] = None, landmark_start: Optional[Tuple]=(1,2), landmark_goal: Optional[Tuple]=(1,1)) -> State:
    """
    Resets the agent at landmark start and sets the agent's goal to landmark_goal.
    The landmarks are given as Tuple representing the positions in the map
    """
    grid = build_grid_world()

    # must call this to include reproduceable stochasticity
    rng = get_gv_rng_if_none(rng)

    agent_position = [Position(*landmark_start)]
    exit_position = [Position(*landmark_goal)]

    # set the agent at a random orientation.
    agent_orientation = choice(rng, list(Orientation))

    grid[exit_position[0]] = Exit()
    agent = Agent(agent_position[0], agent_orientation)
    return State(grid, agent)
    

@reset_function_registry.register
def Agent_1_Goal_1(
    *,
    rng: Optional[rnd.Generator] = None,
) -> State:
    rng = get_gv_rng_if_none(rng)

    # TODO: test creation (e.g. count number of walls, exits, check held item)

    grid = Grid.from_shape((9, 9), factory=Floor)
    # assigns Wall to the border
    draw_wall_boundary(grid)
    draw_area(grid, Area((2, 4), (5,5)), Wall,fill=True)
    draw_area(grid, Area((6, 7),(3, 3)), Wall,fill=True)
    draw_area(grid, Area((4, 4),(2, 7)), Wall,fill=True)
    grid[3,3]=Wall()
    grid[6,7]=Wall()
    grid[4,7]=Floor()
            
    # Define predetermined locations
    agent_positions = [Position(1, 1)]  #positions
    exit_positions = [Position(7, 7)]

    # Randomly select positions
    agent_position = choices(rng, agent_positions,size=1,replace=False)
    exit_position = choices(rng, exit_positions,size=1,replace=False)

    # Ensure they are not the same
    while agent_position[0] == exit_position[0]:
        exit_position = choices(rng, exit_positions,size=1,replace=False)

    # sample agent and exit positions
    #positions = [
    #    position
    #    for position in grid.area.positions()
    #    if isinstance(grid[position], Floor)
    #]
    #agent_position, exit_position = choices(
    #    rng,
    #    positions,
    #    size=2,
    #    replace=False,
    #)

    agent_orientation = choice(rng, list(Orientation))

    grid[exit_position[0]] = Exit()
    agent = Agent(agent_position[0], agent_orientation)
    return State(grid, agent)
    
@reset_function_registry.register
def Agent_1_Goal_3(
    *,
    rng: Optional[rnd.Generator] = None,
) -> State:
    rng = get_gv_rng_if_none(rng)

    # TODO: test creation (e.g. count number of walls, exits, check held item)

    grid = Grid.from_shape((9, 9), factory=Floor)
    # assigns Wall to the border
    draw_wall_boundary(grid)
    draw_area(grid, Area((2, 4), (5,5)), Wall,fill=True)
    draw_area(grid, Area((6, 7),(3, 3)), Wall,fill=True)
    draw_area(grid, Area((4, 4),(2, 7)), Wall,fill=True)
    grid[3,3]=Wall()
    grid[6,7]=Wall()
    grid[4,7]=Floor()
            
    # Define predetermined locations
    agent_positions = [Position(1, 1)]  #positions
    exit_positions = [Position(7, 7),Position(1, 7), Position(7,1)]

    # Randomly select positions
    agent_position = choices(rng, agent_positions,size=1,replace=False)
    exit_position = choices(rng, exit_positions,size=1,replace=False)

    # Ensure they are not the same
    while agent_position[0] == exit_position[0]:
        exit_position = choices(rng, exit_positions,size=1,replace=False)

    # sample agent and exit positions
    #positions = [
    #    position
    #    for position in grid.area.positions()
    #    if isinstance(grid[position], Floor)
    #]
    #agent_position, exit_position = choices(
    #    rng,
    #    positions,
    #    size=2,
    #    replace=False,
    #)

    agent_orientation = choice(rng, list(Orientation))

    grid[exit_position[0]] = Exit()
    agent = Agent(agent_position[0], agent_orientation)
    return State(grid, agent)

@reset_function_registry.register
def Agent_3_Goal_3(
    *,
    rng: Optional[rnd.Generator] = None,
) -> State:
    rng = get_gv_rng_if_none(rng)

    # TODO: test creation (e.g. count number of walls, exits, check held item)

    grid = Grid.from_shape((9, 9), factory=Floor)
    # assigns Wall to the border
    draw_wall_boundary(grid)
    draw_area(grid, Area((2, 4), (5,5)), Wall,fill=True)
    draw_area(grid, Area((6, 7),(3, 3)), Wall,fill=True)
    draw_area(grid, Area((4, 4),(2, 7)), Wall,fill=True)
    grid[3,3]=Wall()
    grid[6,7]=Wall()
    grid[4,7]=Floor()
            
    # Define predetermined locations
    agent_positions = [Position(1, 1),Position(7, 7),Position(1, 7), Position(7,1)]  #positions
    exit_positions = [Position(7, 7),Position(1, 7), Position(7,1)]

    # Randomly select positions
    agent_position = choices(rng, agent_positions,size=1,replace=False)
    exit_position = choices(rng, exit_positions,size=1,replace=False)

    # Ensure they are not the same
    while agent_position[0] == exit_position[0]:
        exit_position = choices(rng, exit_positions,size=1,replace=False)

    # sample agent and exit positions
    #positions = [
    #    position
    #    for position in grid.area.positions()
    #    if isinstance(grid[position], Floor)
    #]
    #agent_position, exit_position = choices(
    #    rng,
    #    positions,
    #    size=2,
    #    replace=False,
    #)

    agent_orientation = choice(rng, list(Orientation))

    grid[exit_position[0]] = Exit()
    agent = Agent(agent_position[0], agent_orientation)
    return State(grid, agent)


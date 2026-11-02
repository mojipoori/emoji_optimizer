from typing import Callable, Iterable, Tuple, Union

import torch

__all__ = ('Betas2', 'OptFloat', 'OptLossClosure', 'Params')

Params = Union[Iterable[torch.Tensor], Iterable[dict]]
Betas2 = Tuple[float, float]
OptFloat = Union[float, None]
OptLossClosure = Union[Callable[[], float], None]
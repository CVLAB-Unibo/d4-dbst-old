from typing import Optional, Tuple

from torch.functional import Tensor

ITEM_T = Tuple[Tensor, Optional[Tensor], Optional[Tensor]]

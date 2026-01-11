from typing import List, Sequence

from deeptrace.common_helper import create_int_range
from deeptrace.processors.modules.expression_restorer.types import ExpressionRestorerArea, ExpressionRestorerModel

expression_restorer_models : List[ExpressionRestorerModel] = [ 'live_portrait' ]

expression_restorer_areas : List[ExpressionRestorerArea] = [ 'upper-face', 'lower-face' ]

expression_restorer_factor_range : Sequence[int] = create_int_range(0, 100, 1)

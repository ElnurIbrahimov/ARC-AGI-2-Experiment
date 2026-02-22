from config.model_config import ModelConfig
from config.dsl_config import (
    DSL_REGISTRY, TOKEN_TO_PRIMITIVE, PRIMITIVE_NAME_TO_ID,
    VOCAB_SIZE, DSLPrimitiveDef, DSLCategory, ArgType,
    PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SEP_TOKEN,
    OPEN_PAREN, CLOSE_PAREN, ARG_SEP, GRID_REF,
    CONST_INT, CONST_COLOR, STRUCTURAL_TOKENS,
    color_to_token, token_to_color, const_to_token, token_to_const,
)
try:
    from config.training_config import TrainingConfig
except ImportError:
    pass

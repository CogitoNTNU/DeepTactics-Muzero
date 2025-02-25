# For environment-specific configurations

# Input settings
OBSERVATION_SHAPE = (3, 84, 84)  # (channels, height, width)
HIDDEN_DIM = 256

# Action space size
ACTION_SIZE = 4  # 🔧 Adjust this when environment details are known

# Residual blocks
NUM_RESIDUAL_BLOCKS = 2

# ================================
# 🔍 Notes for NN Team
# - Adjust OBSERVATION_SHAPE if input format changes.
# - Increase ACTION_SIZE for environments with more discrete actions.
# - Modify NUM_RESIDUAL_BLOCKS based on model complexity.
# ================================

"""Canonical constants for roles, actions, and game parameters."""

# --- Role indices ---
FIGHTER, TANK, MEDIC = 0, 1, 2
F, T, M = FIGHTER, TANK, MEDIC

# --- Role mappings ---
ROLE_MAP = {"FIGHTER": 0, "TANK": 1, "MEDIC": 2, "F": 0, "T": 1, "M": 2}
ROLE_NAMES = {0: "Fighter", 1: "Tank", 2: "Medic"}
ROLE_SHORT = {0: "F", 1: "T", 2: "M"}
ROLE_CHAR_TO_IDX = {"F": 0, "T": 1, "M": 2}
GAME_ROLE_TO_IDX = {"FIGHTER": 0, "TANK": 1, "MEDIC": 2}

# --- Action indices ---
ATTACK, BLOCK, HEAL = 0, 1, 2

# --- Action mappings ---
ACTION_SYMBOLS = {"ATTACK": "A", "BLOCK": "B", "HEAL": "H"}

# --- Plotting ---
ROLE_COLORS = {
    "F": "#e74c3c", "T": "#3498db", "M": "#2ecc71",
    "Fighter": "#e74c3c", "Tank": "#3498db", "Medic": "#2ecc71",
    "FIGHTER": "#e74c3c", "TANK": "#3498db", "MEDIC": "#2ecc71",
    0: "#e74c3c", 1: "#3498db", 2: "#2ecc71",
}
ACTION_COLORS = {
    "ATTACK": "#e74c3c", "BLOCK": "#3498db", "HEAL": "#2ecc71",
    "A": "#e74c3c", "B": "#3498db", "H": "#2ecc71",
}

# --- Game parameters ---
EPSILON = 1e-10
MAX_STAGES = 5
TURNS_PER_STAGE = 2

# --- Known dropout games ---
DROPOUT_GAME_IDS = {
    "01KK14SSY8E64SK69715NN1TMW",   # old dataset dropout
    "01KKZZ4T8F90RB51JW9GHR3B9Q",   # 2 players dropped after R1S1
    "01KKZZ4VRMJT8DA43K6G75XABM",   # 1 player dropped at R4S3
    "01KKZZ54V188BNYZ0WCNNHNC13",   # game never started (batch terminated)
}

# --- Symmetric stat profiles ---
SYMMETRIC_PROFILES = {
    "222_222_222": "all",
    "411_222_222": "last_two",
    "114_222_222": "last_two",
    "141_222_222": "last_two",
}

# --- Role combo <-> env ID mapping ---
ROLE_COMBO_TO_ENV_NUM = {
    "FFF": 82, "FFM": 5189, "FMM": 2712, "FTF": 157,
    "FTM": 139, "MFF": 957, "MMM": 4915, "TFF": 855,
}

# --- All 27 role combos ---
ALL_ROLE_COMBOS = [
    ROLE_SHORT[r0] + ROLE_SHORT[r1] + ROLE_SHORT[r2]
    for r0 in range(3) for r1 in range(3) for r2 in range(3)
]

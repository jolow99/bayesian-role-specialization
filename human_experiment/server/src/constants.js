// Game constants - centralized to avoid duplication across files

// Role identifiers
export const ROLES = {
  FIGHTER: 0,
  TANK: 1,
  MEDIC: 2
};

// Role display names
export const ROLE_NAMES = ["FIGHTER", "TANK", "MEDIC"];

// Action identifiers
export const ACTIONS = {
  ATTACK: 0,
  BLOCK: 1,
  HEAL: 2
};

// Action display names
export const ACTION_NAMES = ["ATTACK", "BLOCK", "HEAL"];

// Stats identifiers
export const STATS = {
  STR: "STR",  // Strength - determines attack damage
  DEF: "DEF",  // Defense - determines damage blocked when blocking
  SUP: "SUP"   // Support - determines healing effectiveness
};

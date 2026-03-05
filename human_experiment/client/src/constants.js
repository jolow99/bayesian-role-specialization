// Timer constants (must match server callbacks.js)
export const STAGE_TIMER_SECONDS = 90;
export const BUFFER_TOTAL_SECONDS = 300;

// Game constants - centralized to avoid duplication across files

// Role identifiers
export const ROLES = {
  FIGHTER: 0,
  TANK: 1,
  MEDIC: 2
};

// Role display names
export const ROLE_NAMES = ["FIGHTER", "TANK", "MEDIC"];

// Role display labels for UI
export const ROLE_LABELS = ["Fighter", "Tank", "Medic"];

// Action identifiers
export const ACTIONS = {
  ATTACK: 0,
  BLOCK: 1,
  HEAL: 2
};

// Action display names
export const ACTION_NAMES = ["ATTACK", "BLOCK", "HEAL"];

// Action icons for UI
export const ACTION_ICONS = {
  ATTACK: "⚔️",
  BLOCK: "🛡️",
  HEAL: "💚"
};

// Role icons for UI
export const ROLE_ICONS = {
  FIGHTER: "🤺",
  TANK: "💂",
  MEDIC: "👩🏻‍⚕️"
};

// Stats identifiers
export const STATS = {
  STR: "STR",  // Strength - determines attack damage
  DEF: "DEF",  // Defense - determines damage blocked when blocking
  SUP: "SUP"   // Support - determines healing effectiveness
};

// Default stat display order
export const STAT_ORDER = ["STR", "DEF", "SUP"];

// Maps role value to its corresponding stat key
// Fighter(0) → STR, Tank(1) → DEF, Medic(2) → SUP
export const ROLE_TO_STAT = ["STR", "DEF", "SUP"];

// Role configurations for UI components
export const ROLE_CONFIG = {
  FIGHTER: {
    label: "Fighter",
    icon: "🤺",
    description: "Attacks most of the time",
    color: "red"
  },
  TANK: {
    label: "Tank",
    icon: "💂",
    description: "Blocks most of the time if the enemy is attacking. Otherwise, acts like a fighter.",
    color: "blue"
  },
  MEDIC: {
    label: "Medic",
    icon: "👩🏻‍⚕️",
    description: "Heals most of the time if the team's health is not full. Otherwise, acts like a fighter.",
    color: "green"
  }
};

// Action configurations for UI components
export const ACTION_CONFIG = {
  ATTACK: {
    label: "Attack",
    icon: "⚔️",
    description: "Deal damage to enemy",
    color: "red"
  },
  BLOCK: {
    label: "Block",
    icon: "🛡️",
    description: "Protect team from enemy attacks",
    color: "blue"
  },
  HEAL: {
    label: "Heal",
    icon: "💚",
    description: "Restore team health",
    color: "green"
  }
};

import { ClassicListenersCollector } from "@empirica/core/admin/classic";
export const Empirica = new ClassicListenersCollector();

// Constants
const ACTIONS = { ATTACK: 0, DEFEND: 1, HEAL: 2 };
const ACTION_NAMES = ["ATTACK", "DEFEND", "HEAL"];
const ROLES = { FIGHTER: 0, TANK: 1, HEALER: 2 };
const ROLE_NAMES = ["FIGHTER", "TANK", "HEALER"];
const ROLE_COMMITMENT_ROUNDS = 3;

// Helper function to generate player stats
function generatePlayerStats(playerId, seed, mode) {
  const rng = seededRandom(seed + playerId);

  if (mode === "balanced") {
    return { STR: 0.33, DEF: 0.33, SUP: 0.34 };

  } else if (mode === "imbalanced-allunique") {
    // Each player strong at different stat: [0.50, 0.25, 0.25]
    const permutations = [
      { STR: 0.50, DEF: 0.25, SUP: 0.25 }, // P0: Strong STR
      { STR: 0.25, DEF: 0.50, SUP: 0.25 }, // P1: Strong DEF
      { STR: 0.25, DEF: 0.25, SUP: 0.50 }, // P2: Strong SUP
    ];
    return permutations[playerId % 3];

  } else if (mode === "imbalanced-oneunique") {
    // Two players same, one different
    const profiles = [
      { STR: 0.25, DEF: 0.25, SUP: 0.50 }, // P0: Strong SUP
      { STR: 0.25, DEF: 0.25, SUP: 0.50 }, // P1: Strong SUP
      { STR: 0.25, DEF: 0.50, SUP: 0.25 }, // P2: Strong DEF (unique)
    ];
    return profiles[playerId % 3];

  } else {
    console.warn(`Unknown statProfile: ${mode}, using balanced`);
    return { STR: 0.33, DEF: 0.33, SUP: 0.34 };
  }
}

// Simple seeded random number generator
function seededRandom(seed) {
  let state = seed;
  return function() {
    state = (state * 9301 + 49297) % 233280;
    return state / 233280;
  };
}

// Convert role to action with probabilistic mapping
function roleToAction(role, gameState, playerStats, rng) {
  const { enemyIntent, teamHealth, maxHealth, epsilon } = gameState;

  let primaryAction;
  switch(role) {
    case ROLES.FIGHTER:
      primaryAction = ACTIONS.ATTACK;
      break;
    case ROLES.TANK:
      primaryAction = (enemyIntent === "WILL_ATTACK") ? ACTIONS.DEFEND : ACTIONS.ATTACK;
      break;
    case ROLES.HEALER:
      primaryAction = (teamHealth <= maxHealth * 0.5) ? ACTIONS.HEAL : ACTIONS.ATTACK;
      break;
    default:
      primaryAction = ACTIONS.ATTACK;
  }

  // With probability epsilon, choose random action
  const eps = epsilon || 0.1;
  if (rng() < eps) {
    return Math.floor(rng() * 3);
  }

  return primaryAction;
}

// Game initialization
Empirica.onGameStart(({ game }) => {
  const treatment = game.get("treatment");
  const {
    statProfile = "balanced",
    maxRounds = 20,
    bossType = "lowDamage",  // NEW: replaces difficulty
    epsilon = 0.1,            // NEW: configurable randomness
    gameSeed = Math.floor(Math.random() * 10000)
  } = treatment;

  // Generate and assign player stats
  game.players.forEach((player, idx) => {
    const stats = generatePlayerStats(idx, gameSeed, statProfile);
    player.set("stats", stats);
    player.set("playerId", idx);
    player.set("actionHistory", []);

    // NEW: Role commitment state
    player.set("currentRole", null);
    player.set("roleStartRound", null);
    player.set("roleEndRound", null);
    player.set("roleHistory", []);
  });

  // Store game settings
  game.set("maxRounds", maxRounds);
  game.set("bossType", bossType);
  game.set("epsilon", epsilon);
  game.set("statProfile", statProfile);
  game.set("gameSeed", gameSeed);
  game.set("maxHealth", 10);
  game.set("initialEnemyHealth", 10);
  game.set("initialTeamHealth", 10);
  game.set("enemyHealth", 10);
  game.set("teamHealth", 10);

  // Calculate boss damage based on bossType
  const maxPlayerDEF = Math.max(...game.players.map(p => p.get("stats").DEF));
  if (bossType === "highDamage") {
    game.set("bossDamage", (maxPlayerDEF * 3) + 1.5);
  } else {
    game.set("bossDamage", (maxPlayerDEF * 3) - 0.7);
  }

  // Create first round
  addGameRound(game, 1);
});

function addGameRound(game, roundNumber) {
  const round = game.addRound({
    name: `Round ${roundNumber}`,
    roundNumber: roundNumber,
  });

  // Initialize round state
  round.set("roundNumber", roundNumber);
  round.set("enemyHealth", game.get("enemyHealth"));
  round.set("teamHealth", game.get("teamHealth"));
  round.set("actions", {});

  // Action selection stage (no timer - wait for all players)
  const actionStage = round.addStage({
    name: "Action Selection",
    duration: 300 // 5 minutes max as safety
  });

  // Store reference to stage on round for early termination
  round.set("actionStage", actionStage);

  // Reveal stage (5 seconds)
  round.addStage({
    name: "Reveal",
    duration: 5
  });
}

Empirica.onRoundStart(({ round }) => {
  const game = round.currentGame;
  const roundNumber = round.get("roundNumber");

  // Reset player actions and handle role commitments
  game.players.forEach(player => {
    player.round.set("action", null);

    // Check if role commitment has expired
    const currentRole = player.get("currentRole");
    const roleEndRound = player.get("roleEndRound");

    if (currentRole !== null && roundNumber > roleEndRound) {
      player.set("currentRole", null);
      player.set("roleStartRound", null);
      player.set("roleEndRound", null);
    }

    // Flag if player needs to select role
    player.round.set("needsRoleSelection", player.get("currentRole") === null);
  });

  // Set enemy intent for this round (50% chance to attack)
  const intent = Math.random() > 0.5 ? "WILL_ATTACK" : "WILL_NOT_ATTACK";
  console.log(`Round ${roundNumber}: Enemy intent set to ${intent}`);
  round.set("enemyIntent", intent);
});

Empirica.onStageStart(({ stage }) => {
  // Empirica automatically advances stages when all players call player.stage.set("submit", true)
  // No additional logic needed here
});

Empirica.onStageEnded(({ stage }) => {
  const round = stage.round;
  const game = round.currentGame;

  if (stage.get("name") === "Action Selection") {
    const gameSeed = game.get("gameSeed");
    const epsilon = game.get("epsilon");
    const teamHealth = game.get("teamHealth");
    const maxHealth = game.get("maxHealth");
    const enemyIntent = round.get("enemyIntent");
    const roundNumber = round.get("roundNumber");

    const actions = [];
    const actionNames = [];
    const roleNames = [];

    game.players.forEach((player, idx) => {
      let currentRole = player.get("currentRole");

      // Check if player just submitted a new role
      const submittedRole = player.round.get("selectedRole");
      if (submittedRole !== null && submittedRole !== undefined) {
        currentRole = submittedRole;

        // Set 3-round commitment
        player.set("currentRole", currentRole);
        player.set("roleStartRound", roundNumber);
        player.set("roleEndRound", roundNumber + ROLE_COMMITMENT_ROUNDS - 1);

        // Log to role history
        const roleHistory = player.get("roleHistory") || [];
        roleHistory.push({
          round: roundNumber,
          role: ROLE_NAMES[currentRole],
          duration: ROLE_COMMITMENT_ROUNDS
        });
        player.set("roleHistory", roleHistory);
      }

      // Default to FIGHTER if no role (shouldn't happen)
      if (currentRole === null || currentRole === undefined) {
        console.warn(`Player ${idx} has no role in round ${roundNumber}, defaulting to FIGHTER`);
        currentRole = ROLES.FIGHTER;
      }

      // Convert role to action
      const rng = seededRandom(gameSeed + roundNumber * 100 + idx);
      const gameState = { enemyIntent, teamHealth, maxHealth, epsilon };
      const action = roleToAction(currentRole, gameState, player.get("stats"), rng);

      actions.push(action);
      actionNames.push(ACTION_NAMES[action]);
      roleNames.push(ROLE_NAMES[currentRole]);
    });

    round.set("actions", actionNames);
    round.set("roles", roleNames);

    // Resolve actions and update health
    resolveActions(game, round, actions);

    // Log the results
    const updatedEnemyHealth = game.get("enemyHealth");
    const updatedTeamHealth = game.get("teamHealth");
    console.log(`After Round ${roundNumber} actions: Enemy HP=${updatedEnemyHealth}, Team HP=${updatedTeamHealth}`);
  }
});

function resolveActions(game, round, actions) {
  // Get player stats
  const stats = game.players.map(p => p.get("stats"));

  const currentEnemyHealth = game.get("enemyHealth");
  const currentTeamHealth = game.get("teamHealth");
  const enemyIntent = round.get("enemyIntent");
  const bossDamage = game.get("bossDamage");

  // Calculate total attack strength
  let totalAttack = 0;
  actions.forEach((action, idx) => {
    if (action === ACTIONS.ATTACK) {
      totalAttack += stats[idx].STR;
    }
  });

  // Calculate max defense (sub-additive: only best defender counts)
  let maxDefense = 0;
  actions.forEach((action, idx) => {
    if (action === ACTIONS.DEFEND) {
      maxDefense = Math.max(maxDefense, stats[idx].DEF);
    }
  });

  // Calculate total healing
  let totalHeal = 0;
  actions.forEach((action, idx) => {
    if (action === ACTIONS.HEAL) {
      totalHeal += stats[idx].SUP;
    }
  });

  // Update enemy health (damage dealt by team)
  // Scale for health out of 10: each point of STR deals ~1.5 damage
  const damageToEnemy = totalAttack * 1.5;
  const newEnemyHealth = Math.max(0, currentEnemyHealth - damageToEnemy);

  // Update team health (damage from enemy minus defense, plus healing)
  let damageToTeam = 0;
  if (enemyIntent === "WILL_ATTACK") {
    const mitigatedDamage = bossDamage - (maxDefense * 3);
    damageToTeam = Math.max(0, mitigatedDamage);
  }

  // Healing scaled for 10 HP: each point of SUP heals ~2 HP
  const healAmount = totalHeal * 2;
  const newTeamHealth = Math.max(0, Math.min(10, currentTeamHealth - damageToTeam + healAmount));

  // Store results in round
  round.set("damageToEnemy", Math.round(damageToEnemy));
  round.set("damageToTeam", Math.round(damageToTeam));
  round.set("healAmount", Math.round(healAmount));
  round.set("newEnemyHealth", Math.round(newEnemyHealth));
  round.set("newTeamHealth", Math.round(newTeamHealth));

  // Update game state
  game.set("enemyHealth", Math.round(newEnemyHealth));
  game.set("teamHealth", Math.round(newTeamHealth));

  // Log action history for each player (their own actions)
  game.players.forEach((player, idx) => {
    const history = player.get("actionHistory") || [];
    history.push({
      round: round.get("roundNumber"),
      action: ACTION_NAMES[actions[idx]],
      enemyHealth: Math.round(newEnemyHealth),
      teamHealth: Math.round(newTeamHealth)
    });
    player.set("actionHistory", history);
  });

  // Also store the full team action history on the game
  const gameHistory = game.get("teamActionHistory") || [];
  gameHistory.push({
    round: round.get("roundNumber"),
    actions: actions.map((action, idx) => ({
      playerId: idx,
      action: ACTION_NAMES[action]
    })),
    enemyHealth: Math.round(newEnemyHealth),
    teamHealth: Math.round(newTeamHealth)
  });
  game.set("teamActionHistory", gameHistory);
}

Empirica.onRoundEnded(({ round }) => {
  const game = round.currentGame;
  const enemyHealth = game.get("enemyHealth");
  const teamHealth = game.get("teamHealth");
  const roundNumber = round.get("roundNumber");
  const maxRounds = game.get("maxRounds");

  console.log(`Round ${roundNumber} ended. Enemy HP=${enemyHealth}, Team HP=${teamHealth}`);

  // Check if game should end
  if (enemyHealth <= 0) {
    game.set("outcome", "WIN");
    console.log("Calling game.end() with WIN");
    game.end("Victory! You defeated the enemy!");
  } else if (teamHealth <= 0) {
    game.set("outcome", "LOSE");
    console.log("Calling game.end() with LOSE");
    game.end("Defeat! Your team was defeated.");
  } else if (roundNumber >= maxRounds) {
    game.set("outcome", "TIMEOUT");
    console.log("Calling game.end() with TIMEOUT");
    game.end("Time's up! The battle has ended.");
  } else {
    // Continue to next round
    console.log(`Creating round ${roundNumber + 1}`);
    addGameRound(game, roundNumber + 1);
  }
});

Empirica.onGameEnded(({ game }) => {
  // Calculate final scores/metrics if needed
  const outcome = game.get("outcome");
  const finalRound = game.get("maxRounds");

  game.players.forEach(player => {
    player.set("finalOutcome", outcome);
  });
});

import { ClassicListenersCollector } from "@empirica/core/admin/classic";
export const Empirica = new ClassicListenersCollector();

// Constants
const ACTIONS = { ATTACK: 0, DEFEND: 1, HEAL: 2 };
const ACTION_NAMES = ["ATTACK", "DEFEND", "HEAL"];
const ROLES = { FIGHTER: 0, TANK: 1, HEALER: 2 };

// Helper function to generate player stats
function generatePlayerStats(playerId, seed, mode) {
  const rng = seededRandom(seed + playerId);

  if (mode === "balanced") {
    return { STR: 0.33, DEF: 0.33, SUP: 0.34 };
  } else if (mode === "specialist") {
    const dominantIdx = playerId % 3;
    const dominantVal = 0.8 + rng() * 0.15;
    const remainder = 1.0 - dominantVal;
    const otherVal = remainder / 2;
    const stats = [otherVal, otherVal, otherVal];
    stats[dominantIdx] = dominantVal;
    return { STR: stats[0], DEF: stats[1], SUP: stats[2] };
  } else {
    // Random mode - Dirichlet-like distribution
    const raw = [rng(), rng(), rng()];
    const sum = raw.reduce((a, b) => a + b, 0);
    return { STR: raw[0] / sum, DEF: raw[1] / sum, SUP: raw[2] / sum };
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

// Game initialization
Empirica.onGameStart(({ game }) => {
  const treatment = game.get("treatment");
  const {
    statProfile = "specialist",
    maxRounds = 20,
    difficulty = 1.0,
    gameSeed = Math.floor(Math.random() * 10000)
  } = treatment;

  // Generate and assign player stats
  game.players.forEach((player, idx) => {
    const stats = generatePlayerStats(idx, gameSeed, statProfile);
    player.set("stats", stats);
    player.set("playerId", idx);
    player.set("actionHistory", []);
    player.set("lastSwitchRound", 0);
  });

  // Store game settings
  game.set("maxRounds", maxRounds);
  game.set("difficulty", difficulty);
  game.set("statProfile", statProfile);
  game.set("gameSeed", gameSeed);
  game.set("maxHealth", 10);
  game.set("initialEnemyHealth", 10);
  game.set("initialTeamHealth", 10);
  game.set("enemyHealth", 10);
  game.set("teamHealth", 10);

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
  // Reset player actions for this round
  round.currentGame.players.forEach(player => {
    player.round.set("action", null);
  });

  // Set enemy intent for this round (70% chance to attack)
  const intent = Math.random() > 0.3 ? "WILL_ATTACK" : "WILL_NOT_ATTACK";
  const roundNumber = round.get("roundNumber");
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
    // Collect all actions
    const actions = [];
    const actionNames = [];
    game.players.forEach(player => {
      const action = player.round.get("action");
      actions.push(action !== null ? action : ACTIONS.ATTACK); // Default to attack if no action
      actionNames.push(action !== null ? ACTION_NAMES[action] : "ATTACK");
    });

    round.set("actions", actionNames);

    // Resolve actions and update health
    resolveActions(game, round, actions);

    // Just log the results after resolving actions
    const enemyHealth = game.get("enemyHealth");
    const teamHealth = game.get("teamHealth");
    const roundNumber = round.get("roundNumber");

    console.log(`After Round ${roundNumber} actions: Enemy HP=${enemyHealth}, Team HP=${teamHealth}`);
  }
});

function resolveActions(game, round, actions) {
  // Get player stats
  const stats = game.players.map(p => p.get("stats"));

  const currentEnemyHealth = game.get("enemyHealth");
  const currentTeamHealth = game.get("teamHealth");
  const enemyIntent = round.get("enemyIntent");
  const difficulty = game.get("difficulty");

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
    // Base enemy damage scaled for 10 HP
    const rawEnemyDamage = 2 * difficulty;
    const mitigatedDamage = rawEnemyDamage - (maxDefense * 3);
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

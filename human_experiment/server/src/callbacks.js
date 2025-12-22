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

// Determine what role a bot should choose based on its strategy
function getBotRoleChoice(player, roundNumber, game) {
  const strategy = player.get("botStrategy");
  if (!strategy) {
    console.warn(`Bot player has no strategy, defaulting to FIGHTER`);
    return ROLES.FIGHTER;
  }

  switch (strategy.type) {
    case "fixed":
      // Always choose the same role
      return strategy.role;

    case "scripted":
      // Follow a predefined sequence of roles
      // strategy.roles is an array like [ROLES.FIGHTER, ROLES.TANK, ...]
      // Use roundNumber to index (1-based, so subtract 1)
      const roleIndex = (roundNumber - 1) % strategy.roles.length;
      return strategy.roles[roleIndex];

    case "random":
      // Choose a random role each time
      const rng = seededRandom(game.get("gameSeed") + roundNumber * 1000 + player.get("playerId"));
      return Math.floor(rng() * 3);

    default:
      console.warn(`Unknown bot strategy type: ${strategy.type}, defaulting to FIGHTER`);
      return ROLES.FIGHTER;
  }
}

// Game initialization
Empirica.onGameStart(({ game }) => {
  const treatment = game.get("treatment");
  const {
    statProfile = "balanced",
    maxRounds = 20,
    bossType = "lowDamage",  // NEW: replaces difficulty
    epsilon = 0.1,            // NEW: configurable randomness
    gameSeed = Math.floor(Math.random() * 10000),
    // Bot configuration
    botPlayers = [],  // Array of bot configs: [{playerId: 0, strategy: {type: "fixed", role: 0}}, ...]
    isTutorial = false,  // Whether this is a tutorial game
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

    // NEW: Bot configuration
    const botConfig = botPlayers.find(b => b.playerId === idx);
    if (botConfig) {
      player.set("isBot", true);
      player.set("botStrategy", botConfig.strategy);
      console.log(`Player ${idx} configured as bot with strategy:`, botConfig.strategy);
    } else {
      player.set("isBot", false);
    }
  });

  // Store game settings
  game.set("maxRounds", maxRounds);
  game.set("bossType", bossType);
  game.set("epsilon", epsilon);
  game.set("statProfile", statProfile);
  game.set("gameSeed", gameSeed);
  game.set("isTutorial", isTutorial);
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

  // Reveal stage (15 seconds to see results)
  // Duration must be in SECONDS (Empirica uses seconds, not milliseconds)
  const revealStage = round.addStage({
    name: "Reveal",
    duration: 15
  });

  console.log(`Created Reveal stage with duration: ${revealStage.get("duration")}`);
}

Empirica.onRoundStart(({ round }) => {
  const game = round.currentGame;
  const roundNumber = round.get("roundNumber");

  // GUARD: Check if this round has already been initialized
  // This prevents double-initialization from multiple callback invocations
  if (round.get("initialized")) {
    console.log(`!!! Round ${roundNumber} already initialized, skipping onRoundStart`);
    return;
  }
  round.set("initialized", true);

  console.log(`=== Round ${roundNumber} Starting ===`);

  // Reset player actions and handle role commitments
  game.players.forEach((player, idx) => {
    player.round.set("action", null);

    // Check if role commitment has expired
    const currentRole = player.get("currentRole");
    const roleStartRound = player.get("roleStartRound");
    const roleEndRound = player.get("roleEndRound");

    console.log(`Player ${idx}: currentRole=${currentRole}, roleStartRound=${roleStartRound}, roleEndRound=${roleEndRound}, roundNumber=${roundNumber}`);

    if (currentRole !== null && roundNumber > roleEndRound) {
      console.log(`Player ${idx}: Role expired, clearing`);
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
  const round = stage.round;
  const game = round.currentGame;
  const stageName = stage.get("name");
  const roundNumber = round.get("roundNumber");
  console.log(`>>> STAGE START: Round ${roundNumber}, Stage: ${stageName}`);

  // Log player submit states
  if (stageName === "Reveal") {
    game.players.forEach((player, idx) => {
      const submitted = player.stage.get("submit");
      console.log(`  Player ${idx} submit state: ${submitted}`);
    });
  }

  // BOT AUTO-PLAY: Have bots automatically select roles and submit
  if (stageName === "Action Selection") {
    game.players.forEach((player, idx) => {
      const isBot = player.get("isBot");
      if (isBot) {
        const currentRole = player.get("currentRole");
        const needsRoleSelection = player.round.get("needsRoleSelection");

        // If bot needs to select a role (commitment expired or first round)
        if (needsRoleSelection || currentRole === null) {
          const roleChoice = getBotRoleChoice(player, roundNumber, game);
          player.round.set("selectedRole", roleChoice);
          console.log(`Bot ${idx} auto-selected role: ${ROLE_NAMES[roleChoice]}`);
        }

        // Bots always submit immediately
        player.stage.set("submit", true);
        console.log(`Bot ${idx} auto-submitted`);
      }
    });
  }

  // Empirica automatically advances stages when all players call player.stage.set("submit", true)
  // No additional logic needed here
});

Empirica.onStageEnded(({ stage }) => {
  const round = stage.round;
  const game = round.currentGame;
  const stageName = stage.get("name");
  const roundNumber = round.get("roundNumber");

  console.log(`<<< STAGE END: Round ${roundNumber}, Stage: ${stageName}`);

  if (stageName === "Action Selection") {
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

  // Store results in round (including previous values for UI)
  round.set("previousEnemyHealth", Math.round(currentEnemyHealth));
  round.set("previousTeamHealth", Math.round(currentTeamHealth));
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

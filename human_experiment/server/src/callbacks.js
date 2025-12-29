import { ClassicListenersCollector } from "@empirica/core/admin/classic";
export const Empirica = new ClassicListenersCollector();

// Constants
const ACTIONS = { ATTACK: 0, DEFEND: 1, HEAL: 2 };
const ACTION_NAMES = ["ATTACK", "DEFEND", "HEAL"];
const ROLES = { FIGHTER: 0, TANK: 1, HEALER: 2 };
const ROLE_NAMES = ["FIGHTER", "TANK", "HEALER"];
const ROLE_COMMITMENT_ROUNDS = 2;

// Helper function to generate player stats
// Stats now sum to 6
function generatePlayerStats(playerId, seed, mode) {
  const rng = seededRandom(seed + playerId);

  if (mode === "balanced") {
    return { STR: 2, DEF: 2, SUP: 2 };

  } else if (mode === "imbalanced-allunique") {
    // Each player strong at different stat: [3, 2, 1]
    const permutations = [
      { STR: 4, DEF: 1, SUP: 1 }, // P0: Strong STR
      { STR: 1, DEF: 4, SUP: 1 }, // P1: Strong DEF
      { STR: 1, DEF: 1, SUP: 4 }, // P2: Strong SUP
    ];
    return permutations[playerId % 3];

  } else if (mode === "imbalanced-oneunique") {
    // Two players same, one different
    const profiles = [
      { STR: 1, DEF: 1, SUP: 4 }, // P0: Strong SUP
      { STR: 1, DEF: 1, SUP: 4 }, // P1: Strong SUP
      { STR: 1, DEF: 4, SUP: 1 }, // P2: Strong DEF (unique)
    ];
    return profiles[playerId % 3];

  } else {
    console.warn(`Unknown statProfile: ${mode}, using balanced`);
    return { STR: 2, DEF: 2, SUP: 2 };
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
  const { enemyIntent, teamHealth, maxHealth, playerDeviateProbability } = gameState;

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

  // With probability playerDeviateProbability, choose random action
  const deviateProb = playerDeviateProbability !== undefined ? playerDeviateProbability : 0.1;
  if (rng() < deviateProb) {
    return Math.floor(rng() * 3);
  }

  return primaryAction;
}

// Determine what role a bot should choose based on its strategy
function getBotRoleChoice(bot, roundNumber, game) {
  const strategy = bot.strategy;
  if (!strategy) {
    console.warn(`Bot has no strategy, defaulting to FIGHTER`);
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
      const rng = seededRandom(game.get("gameSeed") + roundNumber * 1000 + bot.playerId);
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
    playerDeviateProbability = 0.1,  // Probability of random action instead of role-primary action
    enemyAttackProbability = 0.8,    // Probability that enemy will attack each round
    totalPlayers = 3,         // Total number of players (human + bots)
    gameSeed = Math.floor(Math.random() * 10000),
    maxEnemyHealth = 20,      // Maximum enemy health
    maxTeamHealth = 10,       // Maximum team health
    // Bot configuration
    botPlayers = [],  // Array of bot configs: [{playerId: 0, strategy: {type: "fixed", role: 0}}, ...]
    isTutorial = false,  // Whether this is a tutorial game
  } = treatment;

  console.log(`\n===== BOT CONFIGURATION DEBUG =====`);
  console.log(`Game starting with ${game.players.length} human players, target totalPlayers: ${totalPlayers}`);
  console.log(`botPlayers config:`, botPlayers);

  // Store virtual bot configuration on the game
  // Bots are tracked as virtual entities, not as actual player objects
  const virtualBots = [];

  // Create virtual bot entries for each configured bot
  botPlayers.forEach(botConfig => {
    virtualBots.push({
      playerId: botConfig.playerId,
      strategy: botConfig.strategy,
      stats: generatePlayerStats(botConfig.playerId, gameSeed, statProfile),
      currentRole: null,
      roleStartRound: null,
      roleEndRound: null,
      roleHistory: [],
      actionHistory: []
    });
    console.log(`Configured virtual bot at playerId ${botConfig.playerId} with strategy:`, botConfig.strategy);
  });

  game.set("virtualBots", virtualBots);
  game.set("totalPlayers", totalPlayers);

  console.log(`Virtual bots configured: ${virtualBots.length}`);
  console.log(`==============================\n`);

  // Generate and assign player stats for REAL human players
  game.players.forEach((player, idx) => {
    // Find the actual playerId for this player (accounting for virtual bots)
    let actualPlayerId = 0;
    let playersAssigned = 0;

    for (let i = 0; i < totalPlayers; i++) {
      const isBotSlot = virtualBots.some(bot => bot.playerId === i);
      if (!isBotSlot) {
        if (playersAssigned === idx) {
          actualPlayerId = i;
          break;
        }
        playersAssigned++;
      }
    }

    const stats = generatePlayerStats(actualPlayerId, gameSeed, statProfile);
    player.set("stats", stats);
    player.set("playerId", actualPlayerId);
    player.set("actionHistory", []);
    player.set("currentRole", null);
    player.set("roleStartRound", null);
    player.set("roleEndRound", null);
    player.set("roleHistory", []);
    player.set("isBot", false);

    console.log(`Player ${actualPlayerId} (id: ${player.id}) is a human player`);
  });

  // Store game settings
  game.set("maxRounds", maxRounds);
  game.set("bossType", bossType);
  game.set("playerDeviateProbability", playerDeviateProbability);
  game.set("enemyAttackProbability", enemyAttackProbability);
  game.set("statProfile", statProfile);
  game.set("gameSeed", gameSeed);
  game.set("isTutorial", isTutorial);
  game.set("maxHealth", maxTeamHealth);
  game.set("maxEnemyHealth", maxEnemyHealth);
  game.set("initialEnemyHealth", maxEnemyHealth);
  game.set("initialTeamHealth", maxTeamHealth);
  game.set("enemyHealth", maxEnemyHealth);
  game.set("teamHealth", maxTeamHealth);

  // Calculate boss damage based on bossType
  // With stats summing to 6, max DEF is typically 2-3
  // Boss damage should be balanced so defense matters but isn't overwhelming
  if (bossType === "highDamage") {
    game.set("bossDamage", 4);  // High damage boss
  } else {
    game.set("bossDamage", 2);  // Low damage boss
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

  // Reveal stage (8 seconds to see results)
  // Duration must be in SECONDS (Empirica uses seconds, not milliseconds)
  const revealStage = round.addStage({
    name: "Reveal",
    duration: 8
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

  // Reset player actions and handle role commitments for REAL players
  game.players.forEach((player, idx) => {
    player.round.set("action", null);

    // Check if role commitment has expired
    const currentRole = player.get("currentRole");
    const roleStartRound = player.get("roleStartRound");
    const roleEndRound = player.get("roleEndRound");
    const playerId = player.get("playerId");

    console.log(`Player ${playerId}: currentRole=${currentRole}, roleStartRound=${roleStartRound}, roleEndRound=${roleEndRound}, roundNumber=${roundNumber}`);

    if (currentRole !== null && roundNumber > roleEndRound) {
      console.log(`Player ${playerId}: Role expired, clearing`);
      player.set("currentRole", null);
      player.set("roleStartRound", null);
      player.set("roleEndRound", null);
    }

    // Flag if player needs to select role
    player.round.set("needsRoleSelection", player.get("currentRole") === null);
  });

  // Handle virtual bot role commitments
  const virtualBots = game.get("virtualBots") || [];

  // Create new array with updated bots to ensure Empirica detects changes
  const updatedBots = virtualBots.map(bot => {
    if (bot.currentRole !== null && roundNumber > bot.roleEndRound) {
      console.log(`Virtual Bot ${bot.playerId}: Role expired, clearing`);
      return {
        ...bot,
        currentRole: null,
        roleStartRound: null,
        roleEndRound: null
      };
    }
    return bot;
  });

  game.set("virtualBots", updatedBots);

  // Set enemy intent for this round based on enemyAttackProbability
  const enemyAttackProbability = game.get("enemyAttackProbability") || 0.8;
  const intent = Math.random() < enemyAttackProbability ? "WILL_ATTACK" : "WILL_NOT_ATTACK";
  console.log(`Round ${roundNumber}: Enemy intent set to ${intent} (attack probability: ${enemyAttackProbability})`);
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
      const playerId = player.get("playerId");
      console.log(`  Player ${playerId} submit state: ${submitted}`);
    });
  }

  // Handle virtual bot role selection during Action Selection stage
  if (stageName === "Action Selection") {
    const virtualBots = game.get("virtualBots") || [];

    // Create a new array with updated bots to ensure Empirica detects the change
    const updatedBots = virtualBots.map(bot => {
      // If bot needs to select a role (commitment expired or first round)
      if (bot.currentRole === null) {
        const roleChoice = getBotRoleChoice(bot, roundNumber, game);
        console.log(`Virtual Bot ${bot.playerId} auto-selected role: ${ROLE_NAMES[roleChoice]}`);

        return {
          ...bot,
          currentRole: roleChoice,
          roleStartRound: roundNumber,
          roleEndRound: roundNumber + ROLE_COMMITMENT_ROUNDS - 1
        };
      } else {
        console.log(`Virtual Bot ${bot.playerId} keeping role: ${ROLE_NAMES[bot.currentRole]}`);
        return bot;
      }
    });

    game.set("virtualBots", updatedBots);
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
    const playerDeviateProbability = game.get("playerDeviateProbability");
    const teamHealth = game.get("teamHealth");
    const maxHealth = game.get("maxHealth");
    const enemyIntent = round.get("enemyIntent");
    const totalPlayers = game.get("totalPlayers") || 3;
    const virtualBots = game.get("virtualBots") || [];

    console.log(`[onStageEnded] Retrieved ${virtualBots.length} virtual bots from game state`);

    // Build a unified array of all players (real + virtual) indexed by playerId
    const allPlayers = [];
    for (let i = 0; i < totalPlayers; i++) {
      allPlayers[i] = null;
    }

    // Add real players
    game.players.forEach(player => {
      const playerId = player.get("playerId");
      allPlayers[playerId] = { type: "real", player };
      console.log(`[onStageEnded] Added real player at playerId ${playerId}`);
    });

    // Add virtual bots
    virtualBots.forEach(bot => {
      allPlayers[bot.playerId] = { type: "virtual", bot };
      console.log(`[onStageEnded] Added virtual bot at playerId ${bot.playerId}`);
    });

    const actions = [];
    const actionNames = [];
    const roleNames = [];
    const stats = [];

    // Process each player slot
    allPlayers.forEach((entry, playerId) => {
      if (!entry) {
        console.error(`[onStageEnded] ERROR: No player at playerId ${playerId}! (totalPlayers=${totalPlayers}, virtualBots.length=${virtualBots.length}, game.players.length=${game.players.length})`);
        return;
      }

      console.log(`[onStageEnded] Processing playerId ${playerId}, type: ${entry.type}`);

      let currentRole = null;
      let playerStats = null;

      if (entry.type === "real") {
        const player = entry.player;
        currentRole = player.get("currentRole");

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

        playerStats = player.get("stats");
      } else if (entry.type === "virtual") {
        const bot = entry.bot;
        currentRole = bot.currentRole;
        playerStats = bot.stats;
      }

      // Default to FIGHTER if no role (shouldn't happen)
      if (currentRole === null || currentRole === undefined) {
        console.warn(`Player ${playerId} has no role in round ${roundNumber}, defaulting to FIGHTER`);
        currentRole = ROLES.FIGHTER;
      }

      // Convert role to action
      const rng = seededRandom(gameSeed + roundNumber * 100 + playerId);
      const gameState = { enemyIntent, teamHealth, maxHealth, playerDeviateProbability };
      const action = roleToAction(currentRole, gameState, playerStats, rng);

      actions.push(action);
      actionNames.push(ACTION_NAMES[action]);
      roleNames.push(ROLE_NAMES[currentRole]);
      stats.push(playerStats);
    });

    round.set("actions", actionNames);
    round.set("roles", roleNames);

    // Resolve actions and update health
    resolveActions(game, round, actions, stats);

    // Log the results
    const updatedEnemyHealth = game.get("enemyHealth");
    const updatedTeamHealth = game.get("teamHealth");
    console.log(`After Round ${roundNumber} actions: Enemy HP=${updatedEnemyHealth}, Team HP=${updatedTeamHealth}`);
  }
});

function resolveActions(game, round, actions, stats) {
  const currentEnemyHealth = game.get("enemyHealth");
  const currentTeamHealth = game.get("teamHealth");
  const enemyIntent = round.get("enemyIntent");
  const bossDamage = game.get("bossDamage");
  const maxHealth = game.get("maxHealth");

  // Calculate total attack strength (additive)
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

  // Calculate total healing (additive)
  let totalHeal = 0;
  actions.forEach((action, idx) => {
    if (action === ACTIONS.HEAL) {
      totalHeal += stats[idx].SUP;
    }
  });

  // Update enemy health (damage dealt by team)
  // Damage = sum of STR stats
  const damageToEnemy = totalAttack;
  const newEnemyHealth = Math.max(0, currentEnemyHealth - damageToEnemy);

  // Update team health (damage from enemy minus defense, plus healing)
  let damageToTeam = 0;
  if (enemyIntent === "WILL_ATTACK") {
    const mitigatedDamage = bossDamage - maxDefense;
    damageToTeam = Math.max(0, mitigatedDamage);
  }

  // Healing = sum of SUP stats
  const healAmount = totalHeal;
  const newTeamHealth = Math.max(0, Math.min(maxHealth, currentTeamHealth - damageToTeam + healAmount));

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

  // Log action history for each REAL player (their own actions)
  game.players.forEach((player) => {
    const playerId = player.get("playerId");
    const history = player.get("actionHistory") || [];
    history.push({
      round: round.get("roundNumber"),
      action: ACTION_NAMES[actions[playerId]],
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

import { ClassicListenersCollector } from "@empirica/core/admin/classic";
export const Empirica = new ClassicListenersCollector();

// Constants
const ACTIONS = { ATTACK: 0, DEFEND: 1, HEAL: 2 };
const ACTION_NAMES = ["ATTACK", "DEFEND", "HEAL"];
const ROLES = { FIGHTER: 0, TANK: 1, HEALER: 2 };
const ROLE_NAMES = ["FIGHTER", "TANK", "HEALER"];
const TURNS_PER_ROLE = 2; // Each role selection lasts for 2 turns

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
    statProfile,
    totalPlayers,
    maxEnemyHealth,
    maxTeamHealth,
    botPlayers,
  } = treatment;

  // Generate random seed for this game (not from treatment)
  const gameSeed = Math.floor(Math.random() * 10000);

  console.log(`\n===== BOT CONFIGURATION DEBUG =====`);
  console.log(`Game starting with ${game.players.length} human players, target totalPlayers: ${totalPlayers}`);
  console.log(`botPlayers config:`, botPlayers);

  // Create virtual bot entries for each configured bot
  const virtualBots = botPlayers.map(botConfig => ({
    playerId: botConfig.playerId,
    strategy: botConfig.strategy,
    stats: generatePlayerStats(botConfig.playerId, gameSeed, statProfile),
    currentRole: null // Will be set each round during role selection
  }));

  virtualBots.forEach(bot => {
    console.log(`Configured virtual bot at playerId ${bot.playerId} with strategy:`, bot.strategy);
  });

  // Store only dynamic game state and generated values (not treatment config)
  game.set("virtualBots", virtualBots);
  game.set("gameSeed", gameSeed); // Generated at runtime, not from treatment
  game.set("enemyHealth", maxEnemyHealth);
  game.set("teamHealth", maxTeamHealth);

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

    // Set player properties
    player.set("stats", stats);
    player.set("playerId", actualPlayerId);
    player.set("actionHistory", []);
    player.set("roleHistory", []);
    player.set("isBot", false);

    console.log(`Player ${actualPlayerId} (id: ${player.id}) is a human player`);
  });

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

  // Stage 1: Role Selection
  const roleSelectionStage = round.addStage({
    name: "Role Selection",
    duration: 300 // 5 minutes max as safety
  });
  roleSelectionStage.set("stageType", "roleSelection");

  // Stage 2: Turn 1 (actions resolve and results shown)
  const turn1Stage = round.addStage({
    name: "Turn 1",
    duration: 8 // 8 seconds to view results
  });
  turn1Stage.set("stageType", "turn");
  turn1Stage.set("turnNumber", 1);

  // Stage 3: Turn 2 (actions resolve and results shown)
  const turn2Stage = round.addStage({
    name: "Turn 2",
    duration: 8 // 8 seconds to view results
  });
  turn2Stage.set("stageType", "turn");
  turn2Stage.set("turnNumber", 2);

  console.log(`Created Round ${roundNumber} with stages: Role Selection -> Turn 1 (8s) -> Turn 2 (8s)`);
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

  // Reset player round data
  game.players.forEach((player) => {
    player.round.set("selectedRole", null);
  });

  // Set enemy intents for BOTH turns in this round
  const treatment = game.get("treatment");
  const enemyAttackProbability = treatment.enemyAttackProbability;

  const turn1Intent = Math.random() < enemyAttackProbability ? "WILL_ATTACK" : "WILL_NOT_ATTACK";
  const turn2Intent = Math.random() < enemyAttackProbability ? "WILL_ATTACK" : "WILL_NOT_ATTACK";

  round.set("turn1Intent", turn1Intent);
  round.set("turn2Intent", turn2Intent);

  console.log(`Round ${roundNumber}: Turn 1 intent=${turn1Intent}, Turn 2 intent=${turn2Intent}`);
});

Empirica.onStageStart(({ stage }) => {
  const round = stage.round;
  const game = round.currentGame;
  const stageName = stage.get("name");
  const stageType = stage.get("stageType");
  const roundNumber = round.get("roundNumber");
  console.log(`>>> STAGE START: Round ${roundNumber}, Stage: ${stageName} (type: ${stageType})`);

  if (stageType === "roleSelection") {
    // Role Selection stage - handle bot role selection
    const virtualBots = game.get("virtualBots") || [];

    // Create a new array with updated bots to ensure Empirica detects the change
    const updatedBots = virtualBots.map(bot => {
      const roleChoice = getBotRoleChoice(bot, roundNumber, game);
      console.log(`Virtual Bot ${bot.playerId} auto-selected role: ${ROLE_NAMES[roleChoice]}`);

      return {
        ...bot,
        currentRole: roleChoice
      };
    });

    game.set("virtualBots", updatedBots);

    // Store bot roles for this round
    round.set("botRoles", updatedBots.map(bot => ({
      playerId: bot.playerId,
      role: bot.currentRole
    })));
  } else if (stageType === "turn") {
    // Turn stage - resolve actions immediately so results are available to display
    const turnNumber = stage.get("turnNumber");
    console.log(`Turn ${turnNumber} stage starting - resolving actions and showing results for 8 seconds`);

    const treatment = game.get("treatment");
    const gameSeed = game.get("gameSeed");
    const playerDeviateProbability = treatment.playerDeviateProbability;
    const teamHealth = game.get("teamHealth");
    const maxHealth = treatment.maxTeamHealth;
    const enemyIntent = round.get(`turn${turnNumber}Intent`);
    const totalPlayers = treatment.totalPlayers;
    const virtualBots = game.get("virtualBots") || [];

    // Build a unified array of all players (real + virtual) indexed by playerId
    const allPlayers = [];
    for (let i = 0; i < totalPlayers; i++) {
      allPlayers[i] = null;
    }

    // Add real players
    game.players.forEach(player => {
      const playerId = player.get("playerId");
      allPlayers[playerId] = { type: "real", player };
    });

    // Add virtual bots
    virtualBots.forEach(bot => {
      allPlayers[bot.playerId] = { type: "virtual", bot };
    });

    const actions = [];
    const actionNames = [];
    const roleNames = [];
    const stats = [];

    // Process each player slot
    allPlayers.forEach((entry, playerId) => {
      if (!entry) {
        console.error(`ERROR: No player at playerId ${playerId}!`);
        return;
      }

      let currentRole = null;
      let playerStats = null;

      if (entry.type === "real") {
        const player = entry.player;
        // Get role selected during Role Selection stage
        currentRole = player.round.get("selectedRole");
        playerStats = player.get("stats");
      } else if (entry.type === "virtual") {
        const bot = entry.bot;
        currentRole = bot.currentRole;
        playerStats = bot.stats;
      }

      // Default to FIGHTER if no role (shouldn't happen)
      if (currentRole === null || currentRole === undefined) {
        console.warn(`Player ${playerId} has no role in turn ${turnNumber}, defaulting to FIGHTER`);
        currentRole = ROLES.FIGHTER;
      }

      // Convert role to action
      const rng = seededRandom(gameSeed + roundNumber * 1000 + turnNumber * 100 + playerId);
      const gameState = { enemyIntent, teamHealth, maxHealth, playerDeviateProbability };
      const action = roleToAction(currentRole, gameState, playerStats, rng);

      actions.push(action);
      actionNames.push(ACTION_NAMES[action]);
      roleNames.push(ROLE_NAMES[currentRole]);
      stats.push(playerStats);
    });

    // Store turn results on the round
    round.set(`turn${turnNumber}Actions`, actionNames);
    round.set(`turn${turnNumber}Roles`, roleNames);

    // Resolve actions and update health
    resolveTurnActions(game, round, turnNumber, actions, stats, enemyIntent);

    // Log the results
    const updatedEnemyHealth = game.get("enemyHealth");
    const updatedTeamHealth = game.get("teamHealth");
    console.log(`After Turn ${turnNumber}: Enemy HP=${updatedEnemyHealth}, Team HP=${updatedTeamHealth}`);
  }
});

Empirica.onStageEnded(({ stage }) => {
  const round = stage.round;
  const game = round.currentGame;
  const stageName = stage.get("name");
  const stageType = stage.get("stageType");
  const roundNumber = round.get("roundNumber");

  console.log(`<<< STAGE END: Round ${roundNumber}, Stage: ${stageName} (type: ${stageType})`);

  if (stageType === "roleSelection") {
    // Role Selection stage ended - store player roles for this round
    game.players.forEach(player => {
      const submittedRole = player.round.get("selectedRole");
      const playerId = player.get("playerId");

      if (submittedRole !== null && submittedRole !== undefined) {
        // Log to role history
        const roleHistory = player.get("roleHistory") || [];
        roleHistory.push({
          round: roundNumber,
          role: ROLE_NAMES[submittedRole]
        });
        player.set("roleHistory", roleHistory);

        console.log(`Player ${playerId} selected role: ${ROLE_NAMES[submittedRole]}`);
      } else {
        console.warn(`Player ${playerId} did not select a role!`);
      }
    });
  }
  // Turn stages: No action needed here - actions were already resolved in onStageStart
});

function resolveTurnActions(game, round, turnNumber, actions, stats, enemyIntent) {
  const treatment = game.get("treatment");
  const currentEnemyHealth = game.get("enemyHealth");
  const currentTeamHealth = game.get("teamHealth");
  const bossDamage = treatment.bossDamage;
  const maxHealth = treatment.maxTeamHealth;

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
  const damageToEnemy = totalAttack;
  const newEnemyHealth = Math.max(0, currentEnemyHealth - damageToEnemy);

  // Update team health (damage from enemy minus defense, plus healing)
  let damageToTeam = 0;
  if (enemyIntent === "WILL_ATTACK") {
    const mitigatedDamage = bossDamage - maxDefense;
    damageToTeam = Math.max(0, mitigatedDamage);
  }

  const healAmount = totalHeal;
  const newTeamHealth = Math.max(0, Math.min(maxHealth, currentTeamHealth - damageToTeam + healAmount));

  // Store results for this turn on the round
  round.set(`turn${turnNumber}PreviousEnemyHealth`, Math.round(currentEnemyHealth));
  round.set(`turn${turnNumber}PreviousTeamHealth`, Math.round(currentTeamHealth));
  round.set(`turn${turnNumber}DamageToEnemy`, Math.round(damageToEnemy));
  round.set(`turn${turnNumber}DamageToTeam`, Math.round(damageToTeam));
  round.set(`turn${turnNumber}HealAmount`, Math.round(healAmount));
  round.set(`turn${turnNumber}NewEnemyHealth`, Math.round(newEnemyHealth));
  round.set(`turn${turnNumber}NewTeamHealth`, Math.round(newTeamHealth));

  // Update game state
  game.set("enemyHealth", Math.round(newEnemyHealth));
  game.set("teamHealth", Math.round(newTeamHealth));

  // Log action history for each REAL player
  game.players.forEach((player) => {
    const playerId = player.get("playerId");
    const history = player.get("actionHistory") || [];
    history.push({
      round: round.get("roundNumber"),
      turn: turnNumber,
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
    turn: turnNumber,
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
  const treatment = game.get("treatment");
  const enemyHealth = game.get("enemyHealth");
  const teamHealth = game.get("teamHealth");
  const roundNumber = round.get("roundNumber");
  const maxRounds = treatment.maxRounds;

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

  game.players.forEach(player => {
    player.set("finalOutcome", outcome);
  });
});

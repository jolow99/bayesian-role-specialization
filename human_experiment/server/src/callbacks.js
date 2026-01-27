import { ClassicListenersCollector } from "@empirica/core/admin/classic";
import { ACTIONS, ACTION_NAMES, ROLES, ROLE_NAMES } from "./constants.js";
import { readFileSync } from "fs";
import { join } from "path";


// Load round config pools from JSON file
// Use __dirname which is available in bundled Node.js environments (esbuild --platform=node)
const roundConfigPools = JSON.parse(
  readFileSync(join(__dirname, "roundConfigPools.json"), "utf-8")
);

export const Empirica = new ClassicListenersCollector();

// Helper function to get config from pool by ID
function getHumanConfig(configId) {
  const config = roundConfigPools.humanConfigs.find(c => c.id === configId);
  if (!config) {
    console.error(`Human config with ID ${configId} not found!`);
    return null;
  }
  return { ...config, botPlayers: [] }; // Human configs have no bots
}

function getBotConfig(configId) {
  const config = roundConfigPools.botConfigs.find(c => c.id === configId);
  if (!config) {
    console.error(`Bot config with ID ${configId} not found!`);
    return null;
  }
  return config;
}

// Fisher-Yates shuffle with seeded RNG
function shuffleArray(array, rng) {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

// Track lobby player counts and broadcast to waiting players
Empirica.on("player", "introDone", (ctx, { player }) => {
  // Record when player finished intro/tutorial and entered lobby
  const introDoneAt = player.get("introDoneAt");
  if (!introDoneAt) {
    const now = Date.now();
    player.set("introDoneAt", now);
    player.set("lobbyEnteredAt", now);

    // Calculate tutorial duration if we have consentedAt timestamp
    const consentedAt = player.get("consentedAt");
    if (consentedAt) {
      player.set("tutorialDurationMs", now - consentedAt);
    }

    console.log(`Player ${player.id} entered lobby at ${new Date(now).toISOString()}`);
  }

  const game = player.currentGame;
  if (!game || game.hasStarted) {
    return;
  }

  // Update lobby count for this game
  updateLobbyCount(ctx, game);
});

function updateLobbyCount(ctx, game) {
  if (!game || game.hasStarted) {
    return;
  }

  // Get all players in this game who have completed intro
  const allPlayers = Array.from(ctx.scopesByKind("player").values());
  const lobbyPlayers = allPlayers.filter(p => {
    const playerGame = p.currentGame;
    return playerGame &&
           playerGame.id === game.id &&
           p.get("introDone") === true &&
           !game.hasStarted;
  });

  const connectedCount = lobbyPlayers.length;
  const requiredCount = game.get("treatment")?.playerCount || 3;

  console.log(`Lobby update for game ${game.id}: ${connectedCount}/${requiredCount} players ready`);

  // Broadcast count to all waiting players in this game's lobby
  lobbyPlayers.forEach(p => {
    p.set("lobbyPlayersConnected", connectedCount);
    p.set("lobbyPlayersRequired", requiredCount);
  });
}

// Game configuration constants
const TURNS_PER_STAGE = 2; // Each stage (role commitment) lasts for 2 turns

// Helper function to generate player stats
// Stats now sum to 6
function generatePlayerStats(playerId, seed, mode) {

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

  } else if (mode === "imbalanced-oneunique-str") {
    // One player has high STR; others are balanced
    const profiles = [
      { STR: 4, DEF: 1, SUP: 1 }, // P0: Strong STR (unique)
      { STR: 2, DEF: 2, SUP: 2 }, // P1: Balanced
      { STR: 2, DEF: 2, SUP: 2 }, // P2: Balanced
    ];
    return profiles[playerId % 3];

  } else if (mode === "imbalanced-oneunique-def") {
    // One player has high DEF; others are balanced
    const profiles = [
      { STR: 1, DEF: 4, SUP: 1 }, // P0: Strong DEF (unique)
      { STR: 2, DEF: 2, SUP: 2 }, // P1: Balanced
      { STR: 2, DEF: 2, SUP: 2 }, // P2: Balanced
    ];
    return profiles[playerId % 3];

  } else if (mode === "imbalanced-oneunique-sup") {
    // One player has high SUP/HEAL; others are balanced
    const profiles = [
      { STR: 1, DEF: 1, SUP: 4 }, // P0: Strong SUP (unique)
      { STR: 2, DEF: 2, SUP: 2 }, // P1: Balanced
      { STR: 2, DEF: 2, SUP: 2 }, // P2: Balanced
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
      primaryAction = (enemyIntent === "WILL_ATTACK") ? ACTIONS.BLOCK : ACTIONS.ATTACK;
      break;
    case ROLES.MEDIC:
      primaryAction = (teamHealth < maxHealth) ? ACTIONS.HEAL : ACTIONS.ATTACK;
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
    totalPlayers,
    gameSeed,
    humanRoundIds,
    botRoundIds,
  } = treatment;

  const gameStartTime = Date.now();

  console.log(`\n===== GAME START =====`);
  console.log(`Game starting with ${game.players.length} human players, target totalPlayers: ${totalPlayers}`);
  console.log(`Human round IDs: ${humanRoundIds}`);
  console.log(`Bot round IDs per player: ${JSON.stringify(botRoundIds)}`);

  // Store game-level state
  game.set("gameSeed", gameSeed);
  game.set("totalPoints", 0); // Points accumulate across all rounds
  game.set("roundOutcomes", []); // Track win/loss for each round
  game.set("gameStartedAt", gameStartTime); // Timestamp when game started

  // Build shuffled round order using seeded RNG
  // Create round slots: 6 human + 2 bot = 8 rounds
  const roundSlots = [];

  // Add 6 human round slots
  humanRoundIds.forEach((configId, idx) => {
    roundSlots.push({
      type: "human",
      humanConfigId: configId,
      originalIndex: idx
    });
  });

  // Add 2 bot round slots (bot config varies per player, so we just mark the slot index)
  for (let botSlot = 0; botSlot < 2; botSlot++) {
    roundSlots.push({
      type: "bot",
      botSlotIndex: botSlot // 0 or 1, used to look up per-player bot config
    });
  }

  // Shuffle round order using Math.random instead of seeded RNG
  // const shuffleRng = seededRandom(gameSeed);  
  const shuffleRng = Math.random;
  const shuffledRoundOrder = shuffleArray(roundSlots, shuffleRng);

  // Store shuffled order on game for reference
  game.set("shuffledRoundOrder", shuffledRoundOrder);
  console.log(`Before Shuffle:`, roundSlots)
  console.log(`Shuffled round order:`, shuffledRoundOrder.map((r, i) =>
    `Round ${i+1}: ${r.type}${r.type === "human" ? ` (config ${r.humanConfigId})` : ` (slot ${r.botSlotIndex})`}`
  ));

  console.log(`Game seed: ${gameSeed}`);
  console.log(`==============================\n`);

  // Generate and assign permanent player IDs for REAL human players
  // These IDs remain constant throughout the game for consistent P1/P2/P3 labeling
  // Note: Player stats will be regenerated per round based on round config
  game.players.forEach((player, idx) => {
    player.set("actionHistory", []);
    player.set("roleHistory", []);
    player.set("roundOutcomes", []); // Player-specific round outcomes (for accurate bot round display)
    player.set("isBot", false);
    player.set("gamePlayerId", idx); // Permanent player ID (0, 1, or 2)
    player.set("gameStartedAt", gameStartTime); // When this player's game started

    // Store this player's bot config IDs (from the per-player array)
    const playerBotConfigIds = botRoundIds[idx] || botRoundIds[0]; // Fallback to first player's if missing
    player.set("botConfigIds", playerBotConfigIds);
    console.log(`Player ${idx} bot config IDs: ${playerBotConfigIds}`);

    // Calculate lobby wait time
    const lobbyEnteredAt = player.get("lobbyEnteredAt");
    if (lobbyEnteredAt) {
      player.set("lobbyWaitDurationMs", gameStartTime - lobbyEnteredAt);
    }

    console.log(`Player ${idx} (id: ${player.id}) assigned permanent gamePlayerId: ${idx}`);
  });

  // Create first round
  addGameRound(game, 1);
});

function addGameRound(game, roundNumber) {
  console.log(`!!! addGameRound called for round ${roundNumber}`);

  const shuffledRoundOrder = game.get("shuffledRoundOrder");
  const roundSlot = shuffledRoundOrder[roundNumber - 1]; // 0-indexed array

  if (!roundSlot) {
    console.error(`No round slot found for round ${roundNumber}`);
    return;
  }

  const gameSeed = game.get("gameSeed");

  // Determine the round config based on slot type
  let roundConfig;
  let isBotRound = false;

  if (roundSlot.type === "human") {
    // Human round - all players share the same config
    roundConfig = getHumanConfig(roundSlot.humanConfigId);
    console.log(`Round ${roundNumber}: Human round using config ID ${roundSlot.humanConfigId}`);
  } else {
    // Bot round - each player has their own config, but we need a "base" config for round-level properties
    // Use first player's bot config as the base (for round health, etc.)
    const firstPlayer = game.players[0];
    const firstPlayerBotConfigIds = firstPlayer?.get("botConfigIds") || [1, 2];
    const botConfigId = firstPlayerBotConfigIds[roundSlot.botSlotIndex];
    roundConfig = getBotConfig(botConfigId);
    isBotRound = true;
    console.log(`Round ${roundNumber}: Bot round (slot ${roundSlot.botSlotIndex}), base config ID ${botConfigId}`);
  }

  if (!roundConfig) {
    console.error(`No config found for round ${roundNumber}`);
    return;
  }

  const round = game.addRound({
    name: `Round ${roundNumber}`,
    roundNumber: roundNumber,
  });

  console.log(`!!! Created round ${roundNumber} with config:`, roundConfig);

  // Store round number and type
  round.set("roundNumber", roundNumber);
  round.set("stageNumber", 0); // Current stage within this round (0 means not started)
  round.set("isBotRound", isBotRound);
  round.set("roundSlot", roundSlot);

  // Store round config on both game (for server callbacks) and round (for client access)
  game.set(`round${roundNumber}Config`, roundConfig);
  round.set("roundConfig", roundConfig);

  // Store player assignments for this round
  // Human players keep their permanent gamePlayerId (0, 1, 2, etc.)
  const playerAssignments = [];

  game.players.forEach((player) => {
    const gamePlayerId = player.get("gamePlayerId"); // Use permanent ID

    // For bot rounds, each player might have a different config
    let playerRoundConfig = roundConfig;
    if (isBotRound) {
      const playerBotConfigIds = player.get("botConfigIds") || [1, 2];
      const playerBotConfigId = playerBotConfigIds[roundSlot.botSlotIndex];
      playerRoundConfig = getBotConfig(playerBotConfigId);
      // Store player-specific config
      player.set(`round${roundNumber}Config`, playerRoundConfig);
      console.log(`Round ${roundNumber}: Player ${gamePlayerId} using bot config ID ${playerBotConfigId}`);
    }

    // For bot rounds, human always uses stat slot 0 (the imbalanced one); for human rounds, use actual gamePlayerId
    const statSlot = isBotRound ? 0 : gamePlayerId;
    const stats = generatePlayerStats(statSlot, gameSeed + roundNumber * 10000, playerRoundConfig.statProfile);

    playerAssignments.push({
      empiricaPlayerId: player.id,
      playerId: gamePlayerId,
      stats: stats
    });

    console.log(`Round ${roundNumber}: Human Player ${gamePlayerId} stats:`, stats);
  });

  // Store bot configs for this round
  // For bot rounds, each player has their own bot config (stored per-player above)
  // For human rounds, no bots
  let botPositionAssignments = [];
  if (isBotRound) {
    // In bot rounds, we store the bot configs per-player (done in onRoundStart)
    // Here we just mark that this is a bot round
    console.log(`Round ${roundNumber}: Bot round - per-player configs will be set in onRoundStart`);
  }

  game.set(`round${roundNumber}BotAssignments`, botPositionAssignments);
  console.log(`Round ${roundNumber}: Bot configs stored:`, botPositionAssignments.length, "bots (per-player configs in onRoundStart)");

  // Store player assignments on game object with round number key for immediate persistence
  game.set(`round${roundNumber}Assignments`, playerAssignments);
  console.log(`Stored ${playerAssignments.length} player assignments on game for round ${roundNumber}`);

  // Add first stage (role selection + 2 turns)
  addRoundStage(round, 1);
}

function addRoundStage(round, stageNumber) {
  const roundNumber = round.get("roundNumber");
  console.log(`!!! addRoundStage called for round ${roundNumber}, stage ${stageNumber}`);

  // Create a stage for role selection (turns will be resolved automatically when stage starts)
  const stage = round.addStage({
    name: `Stage ${stageNumber}`,
    duration: 300 // 5 minutes max for role selection
  });

  stage.set("stageNumber", stageNumber);

  console.log(`Created Stage ${stageNumber} for Round ${roundNumber}`);
}

Empirica.onRoundStart(({ round }) => {
  const game = round.currentGame;
  const roundNumber = round.get("roundNumber");

  console.log(`!!! onRoundStart called: Round ${roundNumber}, Round ID: ${round.id}`);

  // Idempotency check: only initialize round once even if hook is called multiple times
  const alreadyInitialized = round.get("roundInitialized");
  if (alreadyInitialized) {
    console.log(`Round ${roundNumber} already initialized (hook called multiple times), skipping.`);
    return;
  }
  round.set("roundInitialized", true);

  // Track round start timestamp
  const roundStartTime = Date.now();
  round.set("roundStartedAt", roundStartTime);

  // Set player-round properties now that round is started
  // Retrieve from game-level storage for immediate persistence
  const playerAssignments = game.get(`round${roundNumber}Assignments`) || [];
  if (playerAssignments.length === 0) {
    console.warn(`No player assignments found for round ${roundNumber}`);
  }

  game.players.forEach(player => {
    const assignment = playerAssignments.find(a => a.empiricaPlayerId === player.id);
    if (assignment) {
      player.round.set("stats", assignment.stats);
      player.round.set("playerId", assignment.playerId);
      console.log(`Set player ${assignment.playerId} stats and ID on round ${roundNumber}`);
    } else {
      console.error(`No assignment found for player ${player.id}`);
    }
  });

  // Retrieve from game-level storage for immediate persistence
  const roundConfig = game.get(`round${roundNumber}Config`);

  // Initialize round health from config (do this in onRoundStart for proper persistence)
  round.set("enemyHealth", roundConfig.maxEnemyHealth);
  round.set("teamHealth", roundConfig.maxTeamHealth);
  console.log(`Round ${roundNumber}: Initialized health - Enemy: ${roundConfig.maxEnemyHealth}, Team: ${roundConfig.maxTeamHealth}`);

  // Initialize virtual bots for this round (do this in onRoundStart for proper persistence)
  const gameSeed = game.get("gameSeed");
  const treatment = game.get("treatment");
  const totalPlayers = treatment.totalPlayers;
  // Derive isBotRound from the shuffled round order
  const shuffledRoundOrder = game.get("shuffledRoundOrder");
  const roundSlot = shuffledRoundOrder[roundNumber - 1];
  const isBotRound = roundSlot.type === "bot";

  // For each human player, create their personalized view of virtual bots
  // Each human sees themselves at their gamePlayerId, and bots fill the other positions
  game.players.forEach(player => {
    const humanPlayerId = player.get("gamePlayerId");

    // Get positions NOT occupied by this human (where bots should go)
    const botPositions = [];
    for (let i = 0; i < totalPlayers; i++) {
      if (i !== humanPlayerId) {
        botPositions.push(i);
      }
    }

    // For bot rounds, get this player's specific bot config
    let playerBotConfigs = [];
    let playerRoundConfig = roundConfig;
    if (isBotRound) {
      // Each player has their own bot config stored during addGameRound
      playerRoundConfig = player.get(`round${roundNumber}Config`) || roundConfig;
      playerBotConfigs = playerRoundConfig.botPlayers || [];
    }

    // Assign bots to these positions
    // For bot rounds, bots use stat slots 1 and 2 (balanced), human uses slot 0 (imbalanced)
    const playerVirtualBots = playerBotConfigs.map((botConfig, idx) => ({
      playerId: botPositions[idx], // Assign bot to a non-human position
      botIndex: idx,
      strategy: botConfig.strategy,
      stats: generatePlayerStats(idx + 1, gameSeed + roundNumber * 10000, playerRoundConfig.statProfile),
      currentRole: null // Will be set each stage during role selection
    }));

    // Store per-player virtual bots
    player.round.set("virtualBots", playerVirtualBots);

    // For bot rounds, also initialize per-player health (each player has independent game state)
    if (isBotRound) {
      player.round.set("enemyHealth", playerRoundConfig.maxEnemyHealth);
      player.round.set("teamHealth", playerRoundConfig.maxTeamHealth);
      player.round.set("playerRoundConfig", playerRoundConfig);
      console.log(`Round ${roundNumber}: Player ${humanPlayerId} bot round - Enemy: ${playerRoundConfig.maxEnemyHealth}, Team: ${playerRoundConfig.maxTeamHealth}`);
    }

    console.log(`Round ${roundNumber}: Player ${humanPlayerId} sees ${playerVirtualBots.length} bots at positions:`, playerVirtualBots.map(b => b.playerId));
  });

  // Parse enemy intent sequence from round config (1=attack, 0=rest)
  const enemyIntentSequence = roundConfig.enemyIntentSequence;
  const enemyIntents = enemyIntentSequence.split('').map(char =>
    char === '1' ? "WILL_ATTACK" : "WILL_NOT_ATTACK"
  );

  round.set("enemyIntents", enemyIntents);
  console.log(`Round ${roundNumber}: Pre-generated ${enemyIntents.length} enemy intents:`, enemyIntents);
});

Empirica.onStageStart(({ stage }) => {
  const round = stage.round;
  const game = round.currentGame;
  const roundNumber = round.get("roundNumber");
  const stageNumber = stage.get("stageNumber");

  console.log(`>>> STAGE START: Round ${roundNumber}, Stage ${stageNumber}`);

  // Track stage start timestamp
  const stageStartTime = Date.now();
  stage.set("stageStartedAt", stageStartTime);

  // Bots auto-select roles at stage start (per-player bots only)
  game.players.forEach(player => {
    const playerBots = player.round.get("virtualBots") || [];
    const updatedPlayerBots = playerBots.map(bot => {
      const roleChoice = getBotRoleChoice(bot, roundNumber, game);
      return {
        ...bot,
        currentRole: roleChoice
      };
    });
    player.round.set("virtualBots", updatedPlayerBots);
    console.log(`Player ${player.get("gamePlayerId")} bots updated with roles:`, updatedPlayerBots.map(b => ({ pos: b.playerId, role: ROLE_NAMES[b.currentRole] })));
  });
});

Empirica.onStageEnded(({ stage }) => {
  const round = stage.round;
  const game = round.currentGame;
  const stageNumber = stage.get("stageNumber");
  const roundNumber = round.get("roundNumber");
  const stageType = stage.get("stageType");

  console.log(`<<< STAGE END: Round ${roundNumber}, Stage ${stageNumber}`);

  // Track stage end timestamp and duration
  const stageEndTime = Date.now();
  stage.set("stageEndedAt", stageEndTime);
  const stageStartedAt = stage.get("stageStartedAt");
  if (stageStartedAt) {
    stage.set("stageDurationMs", stageEndTime - stageStartedAt);
  }

  // Handle special stage types (roundEnd, gameEnd)
  if (stageType === "roundEnd") {
    // Round end stage finished - check if we should create next round or end game
    const treatment = game.get("treatment");
    const maxRounds = treatment.maxRounds;

    if (roundNumber < maxRounds) {
      console.log(`Round ${roundNumber} completed. Creating round ${roundNumber + 1}.`);
      addGameRound(game, roundNumber + 1);
    } else {
      // All rounds completed - end the game
      const totalPoints = game.get("totalPoints");
      const roundOutcomes = game.get("roundOutcomes");
      const wins = roundOutcomes.filter(r => r.outcome === "WIN").length;
      const losses = roundOutcomes.filter(r => r.outcome === "LOSE").length;
      const timeouts = roundOutcomes.filter(r => r.outcome === "TIMEOUT").length;

      game.set("finalOutcome", `Game Complete! Wins: ${wins}, Losses: ${losses}, Timeouts: ${timeouts}, Total Points: ${totalPoints}`);
      console.log(`All ${maxRounds} rounds completed. Ending game.`);

      const gameEndStage = round.addStage({
        name: "Game Over",
        duration: 600
      });
      gameEndStage.set("stageType", "gameEnd");
      gameEndStage.set("endMessage", `Game Complete! Total Points: ${totalPoints}`);
    }
    return;
  }

  if (stageType === "gameEnd") {
    // Game end stage finished - actually end the game
    const endMessage = stage.get("endMessage");
    console.log(`Game end stage completed. Ending game.`);
    game.end(endMessage);
    return;
  }

  // Normal game stage ended - players have submitted their roles, resolve both turns
  console.log(`Stage ${stageNumber} role selection ended, resolving turns...`);

  // Update round's current stage number
  round.set("stageNumber", stageNumber);

  // Check if this is a bot round (each player plays independently with bots)
  const shuffledRoundOrder = game.get("shuffledRoundOrder");
  const roundSlot = shuffledRoundOrder[roundNumber - 1];
  const isBotRound = roundSlot.type === "bot";

  if (isBotRound) {
    // Bot round: resolve turns independently for each player
    console.log(`Bot round detected - resolving turns per-player`);
    resolveBothTurnsPerPlayer(game, round, stage, stageNumber);
  } else {
    // Human round: resolve turns with shared state
    resolveBothTurns(game, round, stage, stageNumber);
  }
});

function resolveBothTurns(game, round, stage, stageNumber) {
  const roundNumber = round.get("roundNumber");
  const treatment = game.get("treatment");
  const roundConfig = game.get(`round${roundNumber}Config`);
  const gameSeed = game.get("gameSeed");
  const maxStagesPerRound = treatment.maxStagesPerRound;
  const totalPlayers = treatment.totalPlayers;
  const virtualBots = round.get("virtualBots") || [];
  const enemyIntents = round.get("enemyIntents");

  // Build unified player array
  const allPlayers = [];
  for (let i = 0; i < totalPlayers; i++) {
    allPlayers[i] = null;
  }

  game.players.forEach(player => {
    const playerId = player.round.get("playerId");
    allPlayers[playerId] = { type: "real", player };
  });

  virtualBots.forEach(bot => {
    allPlayers[bot.playerId] = { type: "virtual", bot };
  });

  // Collect player roles and log to history
  game.players.forEach(player => {
    const submittedRole = player.stage.get("selectedRole");
    const roleSubmittedAt = player.stage.get("roleSubmittedAt");
    const playerId = player.round.get("playerId");

    if (submittedRole !== null && submittedRole !== undefined) {
      // Log to role history with timestamp
      const roleHistory = player.get("roleHistory") || [];
      roleHistory.push({
        round: roundNumber,
        stage: stageNumber,
        role: ROLE_NAMES[submittedRole],
        submittedAt: roleSubmittedAt || Date.now()
      });
      player.set("roleHistory", roleHistory);

      console.log(`Player ${playerId} selected role: ${ROLE_NAMES[submittedRole]} for stage ${stageNumber}`);
    } else {
      console.warn(`Player ${playerId} did not select a role!`);
    }
  });

  // Resolve Turn 1 and Turn 2
  const turns = [];
  for (let turnNumber = 1; turnNumber <= 2; turnNumber++) {
    const turnIndex = (stageNumber - 1) * 2 + (turnNumber - 1);
    const enemyIntent = enemyIntents[turnIndex];
    const teamHealth = round.get("teamHealth");
    const maxHealth = roundConfig.maxTeamHealth;
    const playerDeviateProbability = roundConfig.playerDeviateProbability;

    // Determine actions for all players
    const actions = [];
    const actionNames = [];
    const roleNames = [];
    const stats = [];

    allPlayers.forEach((entry, playerId) => {
      if (!entry) {
        console.error(`ERROR: No player at playerId ${playerId}!`);
        return;
      }

      let currentRole = null;
      let playerStats = null;

      if (entry.type === "real") {
        const player = entry.player;
        currentRole = player.stage.get("selectedRole");
        playerStats = player.round.get("stats");
      } else if (entry.type === "virtual") {
        const bot = entry.bot;
        currentRole = bot.currentRole;
        playerStats = bot.stats;
      }

      // Convert role to action
      const rng = seededRandom(gameSeed + roundNumber * 1000 + stageNumber * 100 + turnNumber * 10 + playerId);
      const gameState = { enemyIntent, teamHealth, maxHealth, playerDeviateProbability };
      const action = roleToAction(currentRole, gameState, playerStats, rng);

      actions.push(action);
      actionNames.push(ACTION_NAMES[action]);
      roleNames.push(ROLE_NAMES[currentRole]);
      stats.push(playerStats);
    });

    // Resolve turn and update health
    const turnResult = resolveTurnActions(game, round, stageNumber, turnNumber, actions, stats, enemyIntent);

    turns.push({
      turnNumber,
      actions: actionNames,
      roles: roleNames,
      enemyIntent,
      ...turnResult
    });

    console.log(`After Stage ${stageNumber}, Turn ${turnNumber}: Enemy HP=${turnResult.newEnemyHealth}, Team HP=${turnResult.newTeamHealth}`);

    // Check if round ends after this turn
    // Team losing takes priority - if both hit 0 HP in same turn, team loses
    if (turnResult.newTeamHealth <= 0) {
      // Round lost!
      round.set("outcome", "LOSE");
      round.set("roundEndMessage", "Defeat! Your team was defeated.");
      console.log(`Round ${roundNumber} lost after stage ${stageNumber}, turn ${turnNumber}!`);

      // No points for losing
      const totalTurnsTaken = (stageNumber - 1) * TURNS_PER_STAGE + turnNumber;
      const roundOutcomes = game.get("roundOutcomes");
      const outcomeRecord = { roundNumber, outcome: "LOSE", pointsEarned: 0, turnsTaken: totalTurnsTaken };
      roundOutcomes.push(outcomeRecord);
      game.set("roundOutcomes", roundOutcomes);

      // Also store on each player for consistent client access and update their total points
      game.players.forEach(p => {
        const playerOutcomes = p.get("roundOutcomes") || [];
        playerOutcomes.push(outcomeRecord);
        p.set("roundOutcomes", playerOutcomes);

        // Update player's own cumulative total points (0 for losses)
        // No change needed since pointsEarned is 0
      });

      // Store turn results on round (indexed by stage) for client access
      round.set(`stage${stageNumber}Turns`, turns);
      addRoundEndStage(round);
      return;
    } else if (turnResult.newEnemyHealth <= 0) {
      // Round won!
      round.set("outcome", "WIN");
      round.set("roundEndMessage", "Victory! You defeated the enemy!");
      console.log(`Round ${roundNumber} won after stage ${stageNumber}, turn ${turnNumber}!`);

      // Calculate points for winning: 100 - (100 * T / H)
      // where T = total turns taken, H = max turns per round
      const maxStagesPerRound = game.get("treatment").maxStagesPerRound;
      const maxTurnsPerRound = maxStagesPerRound * TURNS_PER_STAGE; // H = 10
      const totalTurnsTaken = (stageNumber - 1) * TURNS_PER_STAGE + turnNumber; // T
      const pointsEarned = Math.max(0, Math.round(100 - (100 * totalTurnsTaken / maxTurnsPerRound)));

      const currentPoints = game.get("totalPoints");
      game.set("totalPoints", currentPoints + pointsEarned);

      // Record outcome on game level and each player
      const roundOutcomes = game.get("roundOutcomes");
      const outcomeRecord = { roundNumber, outcome: "WIN", pointsEarned, turnsTaken: totalTurnsTaken };
      roundOutcomes.push(outcomeRecord);
      game.set("roundOutcomes", roundOutcomes);

      // Also store on each player for consistent client access and update their total points
      game.players.forEach(p => {
        const playerOutcomes = p.get("roundOutcomes") || [];
        playerOutcomes.push(outcomeRecord);
        p.set("roundOutcomes", playerOutcomes);

        // Update player's own cumulative total points
        const currentPlayerTotal = p.get("totalPoints") || 0;
        p.set("totalPoints", currentPlayerTotal + pointsEarned);
      });

      // Store turn results on round (indexed by stage) for client access
      round.set(`stage${stageNumber}Turns`, turns);
      addRoundEndStage(round);
      return;
    }
  }

  // Both turns completed, round hasn't ended
  // Store turn results on round (indexed by stage) for client access
  round.set(`stage${stageNumber}Turns`, turns);

  // Check if we've hit max stages
  if (stageNumber >= maxStagesPerRound) {
    // Max stages reached - round timeout
    round.set("outcome", "TIMEOUT");
    round.set("roundEndMessage", "Time's up! You couldn't defeat the enemy in time.");
    console.log(`Round ${roundNumber} timed out after max stages!`);

    // No points for timeout
    const totalTurnsTaken = maxStagesPerRound * TURNS_PER_STAGE; // Used all available turns
    const roundOutcomes = game.get("roundOutcomes");
    const outcomeRecord = { roundNumber, outcome: "TIMEOUT", pointsEarned: 0, turnsTaken: totalTurnsTaken };
    roundOutcomes.push(outcomeRecord);
    game.set("roundOutcomes", roundOutcomes);

    // Also store on each player for consistent client access
    // (no points update needed since pointsEarned is 0 for timeout)
    game.players.forEach(p => {
      const playerOutcomes = p.get("roundOutcomes") || [];
      playerOutcomes.push(outcomeRecord);
      p.set("roundOutcomes", playerOutcomes);
    });

    addRoundEndStage(round);
  } else {
    // Continue to next stage
    console.log(`Stage ${stageNumber} complete, round continues. Adding stage ${stageNumber + 1}.`);
    addRoundStage(round, stageNumber + 1);
  }
}

// Per-player turn resolution for bot rounds
// Each player has their own independent game state with bots
function resolveBothTurnsPerPlayer(game, round, _stage, stageNumber) {
  const roundNumber = round.get("roundNumber");
  const treatment = game.get("treatment");
  const baseRoundConfig = game.get(`round${roundNumber}Config`);
  const gameSeed = game.get("gameSeed");
  const maxStagesPerRound = treatment.maxStagesPerRound;
  const totalPlayers = treatment.totalPlayers;
  const enemyIntents = round.get("enemyIntents");

  // Process each player independently
  game.players.forEach(player => {
    const playerId = player.get("gamePlayerId");
    const playerBots = player.round.get("virtualBots") || [];

    // Get player-specific round config (for bot rounds, each player may have different config)
    const playerRoundConfig = player.round.get("playerRoundConfig") || player.get(`round${roundNumber}Config`) || baseRoundConfig;

    // Get or initialize per-player health
    let playerEnemyHealth = player.round.get("enemyHealth");
    let playerTeamHealth = player.round.get("teamHealth");

    // Initialize per-player health on first stage
    if (playerEnemyHealth === null || playerEnemyHealth === undefined) {
      playerEnemyHealth = playerRoundConfig.maxEnemyHealth;
      playerTeamHealth = playerRoundConfig.maxTeamHealth;
    }

    // Check if this player's round already ended
    const playerOutcome = player.round.get("outcome");
    if (playerOutcome) {
      console.log(`Player ${playerId} round already ended with ${playerOutcome}, skipping`);
      return;
    }

    // Build unified player array for this specific player
    const allPlayers = [];
    for (let i = 0; i < totalPlayers; i++) {
      allPlayers[i] = null;
    }

    // Add this player at their position
    allPlayers[playerId] = { type: "real", player };

    // Add this player's bots at their positions
    playerBots.forEach(bot => {
      allPlayers[bot.playerId] = { type: "virtual", bot };
    });

    // Get player's selected role
    const submittedRole = player.stage.get("selectedRole");
    const roleSubmittedAt = player.stage.get("roleSubmittedAt");
    if (submittedRole !== null && submittedRole !== undefined) {
      const roleHistory = player.get("roleHistory") || [];
      roleHistory.push({
        round: roundNumber,
        stage: stageNumber,
        role: ROLE_NAMES[submittedRole],
        submittedAt: roleSubmittedAt || Date.now()
      });
      player.set("roleHistory", roleHistory);
      console.log(`Player ${playerId} selected role: ${ROLE_NAMES[submittedRole]} for stage ${stageNumber}`);
    }

    // Resolve Turn 1 and Turn 2 for this player
    const turns = [];
    for (let turnNumber = 1; turnNumber <= 2; turnNumber++) {
      const turnIndex = (stageNumber - 1) * 2 + (turnNumber - 1);
      const enemyIntent = enemyIntents[turnIndex];
      const maxHealth = playerRoundConfig.maxTeamHealth;
      const playerDeviateProbability = playerRoundConfig.playerDeviateProbability;

      // Determine actions for all players (human + bots)
      const actions = [];
      const actionNames = [];
      const roleNames = [];
      const stats = [];

      allPlayers.forEach((entry, idx) => {
        if (!entry) {
          console.error(`ERROR: No player at position ${idx} for player ${playerId}!`);
          actions.push(ACTIONS.ATTACK);
          actionNames.push(ACTION_NAMES[ACTIONS.ATTACK]);
          roleNames.push("UNKNOWN");
          stats.push({ STR: 2, DEF: 2, SUP: 2 });
          return;
        }

        let currentRole = null;
        let playerStats = null;

        if (entry.type === "real") {
          currentRole = entry.player.stage.get("selectedRole");
          playerStats = entry.player.round.get("stats");
        } else if (entry.type === "virtual") {
          currentRole = entry.bot.currentRole;
          playerStats = entry.bot.stats;
        }

        // Convert role to action
        const rng = seededRandom(gameSeed + roundNumber * 1000 + stageNumber * 100 + turnNumber * 10 + idx);
        const gameState = { enemyIntent, teamHealth: playerTeamHealth, maxHealth, playerDeviateProbability };
        const action = roleToAction(currentRole, gameState, playerStats, rng);

        actions.push(action);
        actionNames.push(ACTION_NAMES[action]);
        roleNames.push(ROLE_NAMES[currentRole]);
        stats.push(playerStats);
      });

      // Resolve turn for this player
      const turnResult = resolveTurnActionsForPlayer(
        player, playerRoundConfig, stageNumber, turnNumber,
        actions, stats, enemyIntent, playerEnemyHealth, playerTeamHealth
      );

      // Update player's health
      playerEnemyHealth = turnResult.newEnemyHealth;
      playerTeamHealth = turnResult.newTeamHealth;
      player.round.set("enemyHealth", playerEnemyHealth);
      player.round.set("teamHealth", playerTeamHealth);

      turns.push({
        turnNumber,
        actions: actionNames,
        roles: roleNames,
        enemyIntent,
        ...turnResult
      });

      console.log(`Player ${playerId} - After Stage ${stageNumber}, Turn ${turnNumber}: Enemy HP=${turnResult.newEnemyHealth}, Team HP=${turnResult.newTeamHealth}`);

      // Check if this player's round ends
      // Team losing takes priority - if both hit 0 HP in same turn, team loses
      if (turnResult.newTeamHealth <= 0) {
        player.round.set("outcome", "LOSE");
        player.round.set("roundEndMessage", "Defeat! Your team was defeated.");
        console.log(`Player ${playerId} lost round ${roundNumber} after stage ${stageNumber}, turn ${turnNumber}!`);

        const totalTurnsTaken = (stageNumber - 1) * TURNS_PER_STAGE + turnNumber;
        player.round.set("pointsEarned", 0);
        player.round.set("turnsTaken", totalTurnsTaken);

        break;
      } else if (turnResult.newEnemyHealth <= 0) {
        player.round.set("outcome", "WIN");
        player.round.set("roundEndMessage", "Victory! You defeated the enemy!");
        console.log(`Player ${playerId} won round ${roundNumber} after stage ${stageNumber}, turn ${turnNumber}!`);

        const maxTurnsPerRound = maxStagesPerRound * TURNS_PER_STAGE;
        const totalTurnsTaken = (stageNumber - 1) * TURNS_PER_STAGE + turnNumber;
        const pointsEarned = Math.max(0, Math.round(100 - (100 * totalTurnsTaken / maxTurnsPerRound)));
        player.round.set("pointsEarned", pointsEarned);
        player.round.set("turnsTaken", totalTurnsTaken);

        break;
      }
    }

    // Store per-player turn results
    player.round.set(`stage${stageNumber}Turns`, turns);

    // Log action history for this player
    const history = player.get("actionHistory") || [];
    turns.forEach(turn => {
      history.push({
        round: roundNumber,
        stage: stageNumber,
        turn: turn.turnNumber,
        action: turn.actions[playerId],
        role: turn.roles[playerId],
        enemyHealth: turn.newEnemyHealth,
        teamHealth: turn.newTeamHealth
      });
    });
    player.set("actionHistory", history);
  });

  // Check if all players have finished their rounds
  const allOutcomes = game.players.map(p => p.round.get("outcome")).filter(o => o);
  if (allOutcomes.length === game.players.length) {
    // All players finished - aggregate results for game tracking
    const wins = allOutcomes.filter(o => o === "WIN").length;
    const losses = allOutcomes.filter(o => o === "LOSE").length;

    // Use majority outcome for round (or first player's outcome)
    const roundOutcome = wins >= losses ? "WIN" : "LOSE";
    round.set("outcome", roundOutcome);

    // Calculate average points for game-level tracking
    let totalPointsEarned = 0;
    game.players.forEach(p => {
      totalPointsEarned += p.round.get("pointsEarned") || 0;
    });
    const avgPoints = Math.round(totalPointsEarned / game.players.length);

    const currentPoints = game.get("totalPoints");
    game.set("totalPoints", currentPoints + avgPoints);

    // Update game-level roundOutcomes with average (for backward compat)
    const roundOutcomes = game.get("roundOutcomes");
    roundOutcomes.push({ roundNumber, outcome: roundOutcome, pointsEarned: avgPoints });
    game.set("roundOutcomes", roundOutcomes);

    // Store player-specific outcomes with their individual turnsTaken and outcome
    // Also update each player's cumulative total points
    game.players.forEach(p => {
      const playerOutcome = p.round.get("outcome");
      const playerPoints = p.round.get("pointsEarned") || 0;
      const playerTurns = p.round.get("turnsTaken") || 0;
      const playerOutcomes = p.get("roundOutcomes") || [];
      playerOutcomes.push({
        roundNumber,
        outcome: playerOutcome,
        pointsEarned: playerPoints,
        turnsTaken: playerTurns
      });
      p.set("roundOutcomes", playerOutcomes);

      // Update player's own cumulative total points
      const currentPlayerTotal = p.get("totalPoints") || 0;
      p.set("totalPoints", currentPlayerTotal + playerPoints);
    });

    round.set("roundEndMessage", roundOutcome === "WIN" ? "Victory!" : "Defeat!");
    addRoundEndStage(round);
  } else if (stageNumber >= maxStagesPerRound) {
    // Max stages reached - timeout for players who haven't finished
    game.players.forEach(p => {
      if (!p.round.get("outcome")) {
        p.round.set("outcome", "TIMEOUT");
        p.round.set("roundEndMessage", "Time's up!");
        p.round.set("pointsEarned", 0);
        p.round.set("turnsTaken", maxStagesPerRound * TURNS_PER_STAGE);
      }
    });

    round.set("outcome", "TIMEOUT");
    round.set("roundEndMessage", "Time's up! You couldn't defeat the enemy in time.");

    // Update game-level roundOutcomes
    const roundOutcomes = game.get("roundOutcomes");
    roundOutcomes.push({ roundNumber, outcome: "TIMEOUT", pointsEarned: 0 });
    game.set("roundOutcomes", roundOutcomes);

    // Store player-specific outcomes and update cumulative points
    game.players.forEach(p => {
      const playerOutcome = p.round.get("outcome");
      const playerPoints = p.round.get("pointsEarned") || 0;
      const playerTurns = p.round.get("turnsTaken") || maxStagesPerRound * TURNS_PER_STAGE;
      const playerOutcomes = p.get("roundOutcomes") || [];
      playerOutcomes.push({
        roundNumber,
        outcome: playerOutcome,
        pointsEarned: playerPoints,
        turnsTaken: playerTurns
      });
      p.set("roundOutcomes", playerOutcomes);

      // Update player's own cumulative total points
      const currentPlayerTotal = p.get("totalPoints") || 0;
      p.set("totalPoints", currentPlayerTotal + playerPoints);
    });

    addRoundEndStage(round);
  } else {
    // Continue to next stage
    console.log(`Stage ${stageNumber} complete for bot round, continuing. Adding stage ${stageNumber + 1}.`);
    addRoundStage(round, stageNumber + 1);
  }
}

// Helper function to resolve turn actions for a specific player
function resolveTurnActionsForPlayer(_player, roundConfig, _stageNumber, _turnNumber, actions, stats, enemyIntent, currentEnemyHealth, currentTeamHealth) {
  const bossDamage = roundConfig.bossDamage;
  const maxHealth = roundConfig.maxTeamHealth;

  // Calculate total attack strength (additive)
  let totalAttack = 0;
  actions.forEach((action, idx) => {
    if (action === ACTIONS.ATTACK) {
      totalAttack += stats[idx].STR;
    }
  });

  // Calculate max defense (sub-additive: only best tank counts)
  let maxDefense = 0;
  actions.forEach((action, idx) => {
    if (action === ACTIONS.BLOCK) {
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

  // Update enemy health
  const damageToEnemy = totalAttack;
  const newEnemyHealth = Math.max(0, currentEnemyHealth - damageToEnemy);

  // Update team health
  let damageToTeam = 0;
  let damageBlocked = 0;
  if (enemyIntent === "WILL_ATTACK") {
    const mitigatedDamage = bossDamage - maxDefense;
    damageToTeam = Math.max(0, mitigatedDamage);
    damageBlocked = Math.min(maxDefense, bossDamage);
  }

  const healAmount = totalHeal;
  const newTeamHealth = Math.max(0, Math.min(maxHealth, currentTeamHealth - damageToTeam + healAmount));

  return {
    previousEnemyHealth: Math.round(currentEnemyHealth),
    previousTeamHealth: Math.round(currentTeamHealth),
    damageToEnemy: Math.round(damageToEnemy),
    damageToTeam: Math.round(damageToTeam),
    damageBlocked: Math.round(damageBlocked),
    healAmount: Math.round(healAmount),
    newEnemyHealth: Math.round(newEnemyHealth),
    newTeamHealth: Math.round(newTeamHealth)
  };
}

function addRoundEndStage(round) {
  const roundNumber = round.get("roundNumber");
  const outcome = round.get("outcome");
  const message = round.get("roundEndMessage");

  console.log(`Adding round end stage with outcome: ${outcome}`);

  const roundEndStage = round.addStage({
    name: "Round Over",
    duration: 600 // 10 minutes max - wait for player to click continue
  });
  roundEndStage.set("stageNumber", "end");
  roundEndStage.set("stageType", "roundEnd");
  roundEndStage.set("endMessage", message);
}

function resolveTurnActions(game, round, stageNumber, turnNumber, actions, stats, enemyIntent) {
  const roundNumber = round.get("roundNumber");
  const roundConfig = game.get(`round${roundNumber}Config`);
  const currentEnemyHealth = round.get("enemyHealth");
  const currentTeamHealth = round.get("teamHealth");
  const bossDamage = roundConfig.bossDamage;
  const maxHealth = roundConfig.maxTeamHealth;

  // Calculate total attack strength (additive)
  let totalAttack = 0;
  actions.forEach((action, idx) => {
    if (action === ACTIONS.ATTACK) {
      totalAttack += stats[idx].STR;
    }
  });

  // Calculate max defense (sub-additive: only best tank counts)
  let maxDefense = 0;
  actions.forEach((action, idx) => {
    if (action === ACTIONS.BLOCK) {
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
  let damageBlocked = 0;
  if (enemyIntent === "WILL_ATTACK") {
    damageToTeam = Math.max(0, bossDamage - maxDefense);
    damageBlocked = Math.min(maxDefense, bossDamage);
  }

  const healAmount = totalHeal;
  const newTeamHealth = Math.max(0, Math.min(maxHealth, currentTeamHealth - damageToTeam + healAmount));

  // Update round state (health persists across turns within the round)
  round.set("enemyHealth", Math.round(newEnemyHealth));
  round.set("teamHealth", Math.round(newTeamHealth));

  // Log action history for each REAL player
  game.players.forEach((player) => {
    const playerId = player.round.get("playerId");
    const history = player.get("actionHistory") || [];

    // Get the player's role for this stage from player.stage
    const currentRole = player.stage.get("selectedRole");

    // Add entry for this turn
    history.push({
      round: roundNumber,
      stage: stageNumber,
      turn: turnNumber,
      action: ACTION_NAMES[actions[playerId]],
      role: ROLE_NAMES[currentRole],
      enemyHealth: Math.round(newEnemyHealth),
      teamHealth: Math.round(newTeamHealth)
    });

    player.set("actionHistory", history);
  });

  // Store team action history on the game for the action history panel
  const gameHistory = game.get("teamActionHistory") || [];
  gameHistory.push({
    round: roundNumber,
    stage: stageNumber,
    turn: turnNumber,
    actions: actions.map((action, idx) => ({
      playerId: idx,
      action: ACTION_NAMES[action]
    })),
    enemyHealth: Math.round(newEnemyHealth),
    teamHealth: Math.round(newTeamHealth),
    enemyIntent: enemyIntent
  });
  game.set("teamActionHistory", gameHistory);

  // Return results for storage on stage
  return {
    previousEnemyHealth: Math.round(currentEnemyHealth),
    previousTeamHealth: Math.round(currentTeamHealth),
    damageToEnemy: Math.round(damageToEnemy),
    damageToTeam: Math.round(damageToTeam),
    damageBlocked: Math.round(damageBlocked),
    healAmount: Math.round(healAmount),
    newEnemyHealth: Math.round(newEnemyHealth),
    newTeamHealth: Math.round(newTeamHealth)
  };
}

Empirica.onRoundEnded(({ round }) => {
  const game = round.currentGame;
  const roundNumber = round.get("roundNumber");
  const outcome = round.get("outcome");

  console.log(`!!! onRoundEnded called: Round ${roundNumber}, Outcome: ${outcome || "ongoing"}`);

  // Track round end timestamp and duration
  const roundEndTime = Date.now();
  round.set("roundEndedAt", roundEndTime);
  const roundStartedAt = round.get("roundStartedAt");
  if (roundStartedAt) {
    round.set("roundDurationMs", roundEndTime - roundStartedAt);
  }

  // Clean up round-specific player data
  game.players.forEach((player) => {
    player.round.set("stats", null);
    player.round.set("playerId", null);
  });

  console.log(`Round ${roundNumber} cleanup complete. Next round will be created from onStageEnded.`);
});

Empirica.onGameEnded(({ game }) => {
  // Calculate final scores/metrics
  const roundOutcomes = game.get("roundOutcomes");
  const finalOutcome = game.get("finalOutcome");

  // Track game end timestamp and total duration
  const gameEndTime = Date.now();
  game.set("gameEndedAt", gameEndTime);
  const gameStartedAt = game.get("gameStartedAt");
  if (gameStartedAt) {
    game.set("gameDurationMs", gameEndTime - gameStartedAt);
  }

  console.log(`Game ended.`);
  console.log(`Round outcomes:`, roundOutcomes);

  // Check if game completed all rounds (vs early termination)
  const treatment = game.get("treatment");
  const maxRounds = treatment?.maxRounds || 0;
  const completedAllRounds = roundOutcomes && roundOutcomes.length >= maxRounds;

  const shuffledRoundOrder = game.get("shuffledRoundOrder") || [];
  const gameSeed = game.get("gameSeed");

  game.players.forEach(player => {
    const playerTotalPoints = player.get("totalPoints") || 0;
    const gamePlayerId = player.get("gamePlayerId");

    // Only mark as "finished" if all rounds were completed (not early termination)
    if (completedAllRounds) {
      player.set("game_complete", true);
    }
    player.set("finalOutcome", finalOutcome);
    player.set("gameEndedAt", gameEndTime);

    // Calculate total experiment duration for this player (from consent to game end)
    const consentedAt = player.get("consentedAt");
    if (consentedAt) {
      player.set("totalExperimentDurationMs", gameEndTime - consentedAt);
    }

    // Build consolidated game summary for clean data export
    // This avoids needing to join across game.csv and player.csv during analysis
    const playerOutcomes = player.get("roundOutcomes") || [];
    const playerBotConfigIds = player.get("botConfigIds") || [];
    const actionHistory = player.get("actionHistory") || [];
    const roleHistory = player.get("roleHistory") || [];

    // Create round results with envId and stats embedded for each round
    const roundResults = playerOutcomes.map((outcome) => {
      const roundNumber = outcome.roundNumber;
      const roundIdx = roundNumber - 1; // 0-indexed
      const roundSlot = shuffledRoundOrder[roundIdx];

      let envId = null;
      let roundType = null;
      let statSlot = null;
      let stats = null;

      if (roundSlot) {
        roundType = roundSlot.type;
        if (roundSlot.type === "human") {
          envId = roundSlot.humanConfigId;
          // Human rounds: player uses their gamePlayerId as stat slot
          statSlot = gamePlayerId;
        } else if (roundSlot.type === "bot") {
          // Map botSlotIndex to this player's specific bot config
          envId = playerBotConfigIds[roundSlot.botSlotIndex];
          // Bot rounds: human always uses stat slot 0 (the imbalanced one)
          statSlot = 0;
        }

        // Regenerate the stats for this round (same logic as addGameRound)
        const roundConfig = game.get(`round${roundNumber}Config`) ||
                           player.get(`round${roundNumber}Config`);
        if (roundConfig) {
          stats = generatePlayerStats(statSlot, gameSeed + roundNumber * 10000, roundConfig.statProfile);
        }
      }

      return {
        roundNumber,
        roundType,
        envId,
        statSlot,
        stats,
        outcome: outcome.outcome,
        pointsEarned: outcome.pointsEarned,
        turnsTaken: outcome.turnsTaken
      };
    });

    const gameSummary = {
      gamePlayerId,
      totalPoints: playerTotalPoints,
      roundResults,
      actionHistory,
      roleHistory
    };

    player.set("gameSummary", gameSummary);

    // Don't overwrite player's individual totalPoints - they already have their own cumulative score
    console.log(`Player ${player.get("gamePlayerId")} final points: ${playerTotalPoints}`);
  });
});

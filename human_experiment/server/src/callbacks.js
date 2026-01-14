import { ClassicListenersCollector } from "@empirica/core/admin/classic";
export const Empirica = new ClassicListenersCollector();

// Constants
const ACTIONS = { ATTACK: 0, BLOCK: 1, HEAL: 2 };
const ACTION_NAMES = ["ATTACK", "BLOCK", "HEAL"];
const ROLES = { FIGHTER: 0, TANK: 1, MEDIC: 2 };
const ROLE_NAMES = ["FIGHTER", "TANK", "MEDIC"];
const TURNS_PER_STAGE = 2; // Each stage (role commitment) lasts for 2 turns

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
      primaryAction = (enemyIntent === "WILL_ATTACK") ? ACTIONS.BLOCK : ACTIONS.ATTACK;
      break;
    case ROLES.MEDIC:
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
    totalPlayers,
    roundConfigs,
  } = treatment;

  // Generate random seed for this game (not from treatment)
  const gameSeed = Math.floor(Math.random() * 10000);

  console.log(`\n===== GAME START =====`);
  console.log(`Game starting with ${game.players.length} human players, target totalPlayers: ${totalPlayers}`);
  console.log(`Game will have ${roundConfigs.length} rounds with different configurations`);

  // Store game-level state
  game.set("gameSeed", gameSeed);
  game.set("totalPoints", 0); // Points accumulate across all rounds
  game.set("roundOutcomes", []); // Track win/loss for each round

  console.log(`Game seed: ${gameSeed}`);
  console.log(`==============================\n`);

  // Generate and assign player stats and IDs for REAL human players
  // Note: Player stats will be regenerated per round based on round config
  game.players.forEach((player, idx) => {
    player.set("actionHistory", []);
    player.set("roleHistory", []);
    player.set("isBot", false);
    console.log(`Player ${idx} (id: ${player.id}) is a human player`);
  });

  // Create first round
  addGameRound(game, 1);
});

function addGameRound(game, roundNumber) {
  console.log(`!!! addGameRound called for round ${roundNumber}`);

  const treatment = game.get("treatment");
  const roundConfigs = treatment.roundConfigs;
  const roundConfig = roundConfigs[roundNumber - 1]; // 0-indexed array

  if (!roundConfig) {
    console.error(`No config found for round ${roundNumber}`);
    return;
  }

  const round = game.addRound({
    name: `Round ${roundNumber}`,
    roundNumber: roundNumber,
  });

  console.log(`!!! Created round ${roundNumber} with config:`, roundConfig);

  // Store round number (minimal data that must be set here)
  round.set("roundNumber", roundNumber);
  round.set("stageNumber", 0); // Current stage within this round (0 means not started)

  // Store round config on both game (for server callbacks) and round (for client access)
  game.set(`round${roundNumber}Config`, roundConfig);
  round.set("roundConfig", roundConfig);

  // Note: virtualBots will be initialized in onRoundStart for proper persistence
  const totalPlayers = treatment.totalPlayers;
  const botPlayers = roundConfig.botPlayers || [];
  const gameSeed = game.get("gameSeed");

  console.log(`Round ${roundNumber}: Will configure ${botPlayers.length} virtual bots in onRoundStart`);

  // Store player assignments for this round
  const playerAssignments = [];
  game.players.forEach((player, idx) => {
    // Find the actual playerId for this player (accounting for virtual bots)
    let actualPlayerId = 0;
    let playersAssigned = 0;

    for (let i = 0; i < totalPlayers; i++) {
      // Check if this slot is taken by a bot using botPlayers config
      const isBotSlot = botPlayers.some(botConfig => botConfig.playerId === i);
      if (!isBotSlot) {
        if (playersAssigned === idx) {
          actualPlayerId = i;
          break;
        }
        playersAssigned++;
      }
    }

    const stats = generatePlayerStats(actualPlayerId, gameSeed + roundNumber * 10000, roundConfig.statProfile);

    playerAssignments.push({
      empiricaPlayerId: player.id,
      playerId: actualPlayerId,
      stats: stats
    });

    console.log(`Round ${roundNumber}: Player ${actualPlayerId} stats:`, stats);
  });

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
  const botPlayers = roundConfig.botPlayers || [];
  const gameSeed = game.get("gameSeed");

  const virtualBots = botPlayers.map(botConfig => ({
    playerId: botConfig.playerId,
    strategy: botConfig.strategy,
    stats: generatePlayerStats(botConfig.playerId, gameSeed + roundNumber * 10000, roundConfig.statProfile),
    currentRole: null // Will be set each stage during role selection
  }));

  round.set("virtualBots", virtualBots);
  console.log(`Round ${roundNumber}: Initialized ${virtualBots.length} virtual bots`);

  const enemyAttackProbability = roundConfig.enemyAttackProbability;
  const maxStagesPerRound = game.get("treatment").maxStagesPerRound;

  // Pre-generate enemy intents for all possible turns in this round
  // Each stage has 2 turns, so we need up to maxStagesPerRound * 2 intents
  const enemyIntents = [];
  for (let i = 0; i < maxStagesPerRound * 2; i++) {
    const intent = Math.random() < enemyAttackProbability ? "WILL_ATTACK" : "WILL_NOT_ATTACK";
    enemyIntents.push(intent);
  }

  round.set("enemyIntents", enemyIntents);
  console.log(`Round ${roundNumber}: Pre-generated ${enemyIntents.length} enemy intents:`, enemyIntents);
});

Empirica.onStageStart(({ stage }) => {
  const round = stage.round;
  const game = round.currentGame;
  const roundNumber = round.get("roundNumber");
  const stageNumber = stage.get("stageNumber");

  console.log(`>>> STAGE START: Round ${roundNumber}, Stage ${stageNumber}`);

  // Bots auto-select roles at stage start
  const virtualBots = round.get("virtualBots") || [];
  const updatedBots = virtualBots.map(bot => {
    const roleChoice = getBotRoleChoice(bot, roundNumber, game);
    console.log(`Virtual Bot ${bot.playerId} auto-selected role: ${ROLE_NAMES[roleChoice]}`);
    return {
      ...bot,
      currentRole: roleChoice
    };
  });
  round.set("virtualBots", updatedBots);

  // Store bot roles on stage for client access
  stage.set("botRoles", updatedBots.map(bot => ({
    playerId: bot.playerId,
    role: bot.currentRole
  })));
});

Empirica.onStageEnded(({ stage }) => {
  const round = stage.round;
  const game = round.currentGame;
  const stageNumber = stage.get("stageNumber");
  const roundNumber = round.get("roundNumber");
  const stageType = stage.get("stageType");

  console.log(`<<< STAGE END: Round ${roundNumber}, Stage ${stageNumber}`);

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

  // Resolve both turns for this stage
  resolveBothTurns(game, round, stage, stageNumber);
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
    const playerId = player.round.get("playerId");

    if (submittedRole !== null && submittedRole !== undefined) {
      // Log to role history
      const roleHistory = player.get("roleHistory") || [];
      roleHistory.push({
        round: roundNumber,
        stage: stageNumber,
        role: ROLE_NAMES[submittedRole]
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
    if (turnResult.newEnemyHealth <= 0) {
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

      // Record outcome
      const roundOutcomes = game.get("roundOutcomes");
      roundOutcomes.push({ roundNumber, outcome: "WIN", pointsEarned, turnsTaken: totalTurnsTaken });
      game.set("roundOutcomes", roundOutcomes);

      // Store turn results on round (indexed by stage) for client access
      round.set(`stage${stageNumber}Turns`, turns);
      addRoundEndStage(round);
      return;
    } else if (turnResult.newTeamHealth <= 0) {
      // Round lost!
      round.set("outcome", "LOSE");
      round.set("roundEndMessage", "Defeat! Your team was defeated.");
      console.log(`Round ${roundNumber} lost after stage ${stageNumber}, turn ${turnNumber}!`);

      // No points for losing
      const totalTurnsTaken = (stageNumber - 1) * TURNS_PER_STAGE + turnNumber;
      const roundOutcomes = game.get("roundOutcomes");
      roundOutcomes.push({ roundNumber, outcome: "LOSE", pointsEarned: 0, turnsTaken: totalTurnsTaken });
      game.set("roundOutcomes", roundOutcomes);

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
    roundOutcomes.push({ roundNumber, outcome: "TIMEOUT", pointsEarned: 0, turnsTaken: totalTurnsTaken });
    game.set("roundOutcomes", roundOutcomes);

    addRoundEndStage(round);
  } else {
    // Continue to next stage
    console.log(`Stage ${stageNumber} complete, round continues. Adding stage ${stageNumber + 1}.`);
    addRoundStage(round, stageNumber + 1);
  }
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
  if (enemyIntent === "WILL_ATTACK") {
    const mitigatedDamage = bossDamage - maxDefense;
    damageToTeam = Math.max(0, mitigatedDamage);
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

  // Clean up round-specific player data
  game.players.forEach((player) => {
    player.round.set("stats", null);
    player.round.set("playerId", null);
  });

  console.log(`Round ${roundNumber} cleanup complete. Next round will be created from onStageEnded.`);
});

Empirica.onGameEnded(({ game }) => {
  // Calculate final scores/metrics
  const totalPoints = game.get("totalPoints");
  const roundOutcomes = game.get("roundOutcomes");
  const finalOutcome = game.get("finalOutcome");

  console.log(`Game ended. Total points: ${totalPoints}`);
  console.log(`Round outcomes:`, roundOutcomes);

  game.players.forEach(player => {
    player.set("finalOutcome", finalOutcome);
    player.set("totalPoints", totalPoints);
  });
});

import React, { useState, useEffect } from "react";
import { MockDataProvider } from "../components/tutorial";
import { BattleField } from "../components/BattleField";
import { ActionMenu } from "../components/ActionMenu";
import { ResultsPanel } from "../components/ResultsPanel";
import { ActionHistory } from "../components/ActionHistory";

const ROLES = { FIGHTER: 0, TANK: 1, HEALER: 2 };
const ROLE_NAMES = ["Fighter", "Tank", "Healer"];

export function Tutorial2({ next }) {
  const [selectedRole, setSelectedRole] = useState(null);
  const [currentRound, setCurrentRound] = useState(2); // Start directly at role selection (2)
  const [roundResults, setRoundResults] = useState([]);
  const [mockData, setMockData] = useState(null);
  const [showOutcome, setShowOutcome] = useState(false);
  const [outcome, setOutcome] = useState(null);
  const [showDamageAnimation, setShowDamageAnimation] = useState(false);

  // Bot players: One Tank (defends when enemy attacks), One Healer (heals when health < 50%)
  const actualBotRoles = [ROLES.TANK, ROLES.HEALER];

  useEffect(() => {
    // Initialize directly at role selection with Round 1 already completed
    const round1Result = simulateRound(1, null, 10, 10); // null = player hasn't chosen yet
    setRoundResults([round1Result]);

    const roleSelectionMockData = createMockDataForRoleSelection(round1Result);
    setMockData(roleSelectionMockData);
  }, []);

  const getBotAction = (role, enemyAttacks, teamHealth) => {
    if (role === ROLES.FIGHTER) return "ATTACK";
    if (role === ROLES.TANK) return enemyAttacks ? "DEFEND" : "ATTACK";
    if (role === ROLES.HEALER) {
      // Healer heals when team health < 50%, otherwise attacks
      return teamHealth < 5 ? "HEAL" : "ATTACK";
    }
    return "ATTACK";
  };

  // Simulate a single turn of combat using real game stats
  const simulateTurn = (turnNum, playerRole, currentEnemyHP, currentTeamHP, enemyAttacks, forcedActions = null) => {
    const enemyIntent = enemyAttacks ? "WILL_ATTACK" : "WILL_REST";
    const STR = 2, DEF = 2, SUP = 2; // Real game stats
    const bossDamage = 2; // From treatments.yaml
    const maxTeamHealth = 10;

    let bot1Action, bot2Action;

    // Use forced actions if provided, otherwise calculate
    if (forcedActions) {
      bot1Action = forcedActions[0];
      bot2Action = forcedActions[1];
    } else {
      bot1Action = getBotAction(actualBotRoles[0], enemyAttacks, currentTeamHP);
      bot2Action = getBotAction(actualBotRoles[1], enemyAttacks, currentTeamHP);
    }

    let playerAction, actions, roles, stats;
    if (playerRole === null) {
      // Player hasn't chosen yet
      playerAction = null;
      actions = [bot1Action, bot2Action];
      roles = actualBotRoles;
      stats = [{ STR, DEF, SUP }, { STR, DEF, SUP }];
    } else {
      if (playerRole === ROLES.FIGHTER) {
        playerAction = "ATTACK";
      } else if (playerRole === ROLES.TANK) {
        playerAction = enemyAttacks ? "DEFEND" : "ATTACK";
      } else if (playerRole === ROLES.HEALER) {
        playerAction = currentTeamHP <= 5 ? "HEAL" : "ATTACK";
      } else {
        playerAction = "ATTACK";
      }
      actions = [bot1Action, bot2Action, playerAction];
      roles = [...actualBotRoles, playerRole];
      stats = [{ STR, DEF, SUP }, { STR, DEF, SUP }, { STR, DEF, SUP }];
    }

    // Calculate damage using real game logic (additive STR)
    let totalAttack = 0;
    actions.forEach((action, idx) => {
      if (action === "ATTACK") {
        totalAttack += stats[idx].STR;
      }
    });
    const damageToEnemy = totalAttack;

    // Calculate defense using real game logic (max DEF, not additive)
    let maxDefense = 0;
    actions.forEach((action, idx) => {
      if (action === "DEFEND") {
        maxDefense = Math.max(maxDefense, stats[idx].DEF);
      }
    });

    let damageToTeam = 0;
    if (enemyAttacks) {
      const mitigatedDamage = bossDamage - maxDefense;
      damageToTeam = Math.max(0, mitigatedDamage);
    }

    // Calculate healing using real game logic (additive SUP)
    let totalHeal = 0;
    actions.forEach((action, idx) => {
      if (action === "HEAL") {
        totalHeal += stats[idx].SUP;
      }
    });
    const healAmount = totalHeal;

    const newEnemyHP = Math.max(0, currentEnemyHP - damageToEnemy);
    const newTeamHP = Math.max(0, Math.min(maxTeamHealth, currentTeamHP - damageToTeam + healAmount));

    return {
      turnNum,
      enemyAttacks,
      enemyIntent,
      bot1Action,
      bot2Action,
      playerAction,
      playerRole,
      actions,
      roles,
      damageToEnemy: Math.round(damageToEnemy),
      damageToTeam: Math.round(damageToTeam),
      healAmount: Math.round(healAmount),
      enemyHealth: Math.round(newEnemyHP),
      teamHealth: Math.round(newTeamHP),
      previousEnemyHealth: Math.round(currentEnemyHP),
      previousTeamHealth: Math.round(currentTeamHP)
    };
  };

  // Simulate a complete round with 2 turns
  const simulateRound = (roundNum, playerRole, startEnemyHP, startTeamHP) => {
    // Turn 1: Tank defends, Healer heals, Enemy attacks
    const turn1 = simulateTurn(1, playerRole, startEnemyHP, startTeamHP, true, ["DEFEND", "HEAL"]);

    // Turn 2: Both bots attack, Enemy rests
    const turn2 = simulateTurn(2, playerRole, turn1.enemyHealth, turn1.teamHealth, false, ["ATTACK", "ATTACK"]);

    return {
      roundNum,
      turns: [turn1, turn2],
      enemyHealth: turn2.enemyHealth,
      teamHealth: turn2.teamHealth,
      // For backward compatibility, expose the last turn's data at the top level
      enemyIntent: turn2.enemyIntent,
      actions: turn2.actions,
      roles: turn2.roles,
      damageToEnemy: turn2.damageToEnemy,
      damageToTeam: turn2.damageToTeam,
      healAmount: turn2.healAmount,
      previousEnemyHealth: turn2.previousEnemyHealth,
      previousTeamHealth: turn2.previousTeamHealth
    };
  };

  const handleRoleSelect = (role) => {
    setSelectedRole(role);
  };

  const handleSubmit = () => {
    if (selectedRole === null) return;

    // Simulate rounds 2 and 3 with selected role
    const results = [...roundResults];
    let currentEnemyHP = roundResults[0].enemyHealth;
    let currentTeamHP = roundResults[0].teamHealth;

    for (let i = 2; i <= 3; i++) {
      const result = simulateRound(i, selectedRole, currentEnemyHP, currentTeamHP);
      results.push(result);
      currentEnemyHP = result.enemyHealth;
      currentTeamHP = result.teamHealth;

      if (currentEnemyHP <= 0 || currentTeamHP <= 0) {
        break;
      }
    }

    setRoundResults(results);

    // Determine outcome - non-normative, let users make their own judgment
    let outcomeMessage;
    if (currentEnemyHP <= 0) {
      outcomeMessage = "The enemy was defeated!";
    } else if (currentTeamHP <= 0) {
      outcomeMessage = "The team was defeated.";
    } else {
      outcomeMessage = "The battle concluded after 3 rounds.";
    }

    setOutcome({
      message: outcomeMessage,
      enemyHealth: currentEnemyHP,
      teamHealth: currentTeamHP
    });

    // Show round 2
    const round2Result = results[1];
    const round2MockData = createMockDataForRound(round2Result, 2, true);
    setMockData(round2MockData);
    setCurrentRound(3);
  };

  // Trigger damage animation when advancing rounds
  useEffect(() => {
    if (currentRound === 1 || currentRound >= 3) {
      setShowDamageAnimation(true);
      const timer = setTimeout(() => setShowDamageAnimation(false), 3000);
      return () => clearTimeout(timer);
    }
  }, [currentRound]);

  const handleNextRound = () => {
    const nextRoundIndex = currentRound - 1; // currentRound 3 = roundResults[1] (round 2)
    if (nextRoundIndex < roundResults.length) {
      const nextRoundResult = roundResults[nextRoundIndex];
      const nextRoundMockData = createMockDataForRound(nextRoundResult, nextRoundIndex + 1, true);
      setMockData(nextRoundMockData);
      setCurrentRound(currentRound + 1);
    } else {
      // Show outcome overlay
      setShowOutcome(true);
    }
  };

  const createMockDataForRoleSelection = (round1Result) => {
    const players = [
      { id: "bot-1", playerId: 0, stats: { STR: 2, DEF: 2, SUP: 2 } },
      { id: "bot-2", playerId: 1, stats: { STR: 2, DEF: 2, SUP: 2 } },
      { id: "tutorial-player", playerId: 2, stats: { STR: 2, DEF: 2, SUP: 2 } }
    ];

    // Build team history from round 1 turns
    const teamHistory = round1Result.turns ? round1Result.turns.map(turn => ({
      round: 1,
      turn: turn.turnNum,
      enemyHealth: turn.enemyHealth,
      teamHealth: turn.teamHealth,
      enemyIntent: turn.enemyIntent,
      actions: turn.actions.map((action, idx) => ({
        playerId: idx,
        action: action
      }))
    })) : [];

    return {
      game: {
        enemyHealth: round1Result.enemyHealth,
        maxEnemyHealth: 10,
        teamHealth: round1Result.teamHealth,
        maxTeamHealth: 10,
        treatment: {
          totalPlayers: 3,
          maxRounds: 3,
          maxEnemyHealth: 10,
          maxTeamHealth: 10
        }
      },
      player: {
        id: "tutorial-player",
        playerId: 2,
        stats: { STR: 2, DEF: 2, SUP: 2 },
        roleOrder: [ROLES.FIGHTER, ROLES.TANK, ROLES.HEALER],
        stage: {},
        round: {}
      },
      players: players,
      round: {
        roundNumber: 2
      },
      stage: {
        name: "roleSelection",
        stageType: "roleSelection"
      },
      teamHistory: teamHistory
    };
  };

  const createMockDataForRound = (roundResult, roundNum, includePlayer) => {
    const players = includePlayer
      ? [
          { id: "bot-1", playerId: 0, stats: { STR: 2, DEF: 2, SUP: 2 } },
          { id: "bot-2", playerId: 1, stats: { STR: 2, DEF: 2, SUP: 2 } },
          { id: "tutorial-player", playerId: 2, stats: { STR: 2, DEF: 2, SUP: 2 } }
        ]
      : [
          { id: "bot-1", playerId: 0, stats: { STR: 2, DEF: 2, SUP: 2 } },
          { id: "bot-2", playerId: 1, stats: { STR: 2, DEF: 2, SUP: 2 } }
        ];

    // Build team history from the turns in this round
    const teamHistory = roundResult.turns ? roundResult.turns.map(turn => ({
      round: roundNum,
      turn: turn.turnNum,
      enemyHealth: turn.enemyHealth,
      teamHealth: turn.teamHealth,
      enemyIntent: turn.enemyIntent,
      actions: turn.actions.map((action, idx) => ({
        playerId: idx,
        action: action
      }))
    })) : [];

    return {
      game: {
        enemyHealth: roundResult.enemyHealth,
        maxEnemyHealth: 10,
        teamHealth: roundResult.teamHealth,
        maxTeamHealth: 10,
        treatment: {
          totalPlayers: includePlayer ? 3 : 2,
          maxRounds: 3,
          maxEnemyHealth: 10,
          maxTeamHealth: 10
        }
      },
      player: {
        id: "tutorial-player",
        playerId: 2,
        stats: { STR: 2, DEF: 2, SUP: 2 },
        roleOrder: [ROLES.FIGHTER, ROLES.TANK, ROLES.HEALER],
        stage: {},
        round: {}
      },
      players: players,
      round: {
        roundNumber: roundNum,
        [`turn1Intent`]: roundResult.enemyIntent,
        [`turn1Actions`]: roundResult.actions,
        [`turn1Roles`]: roundResult.roles,
        [`turn1DamageToEnemy`]: roundResult.damageToEnemy,
        [`turn1DamageToTeam`]: roundResult.damageToTeam,
        [`turn1HealAmount`]: roundResult.healAmount,
        [`turn1PreviousEnemyHealth`]: roundResult.previousEnemyHealth,
        [`turn1PreviousTeamHealth`]: roundResult.previousTeamHealth
      },
      stage: {
        name: `turn1`,
        stageType: "turn",
        turnNumber: 1
      },
      teamHistory: teamHistory
    };
  };

  const buildAllPlayers = (includePlayer) => {
    if (!mockData) return [];
    return mockData.players.map((p, idx) => ({
      type: includePlayer && idx === 2 ? "real" : "virtual",
      player: p,
      playerId: p.playerId,
      bot: (includePlayer && idx === 2) ? null : { stats: p.stats, playerId: p.playerId }
    }));
  };

  const handlePlayAgain = () => {
    setSelectedRole(null);
    setCurrentRound(2);
    setShowOutcome(false);
    setOutcome(null);

    // Re-initialize at role selection
    const round1Result = simulateRound(1, null, 10, 10);
    setRoundResults([round1Result]);

    const roleSelectionMockData = createMockDataForRoleSelection(round1Result);
    setMockData(roleSelectionMockData);
  };

  const handleStartMainGame = () => {
    next();
  };

  if (!mockData) return null;

  const isRoleSelection = currentRound === 2;
  const isTurnStage = currentRound >= 3;
  const currentRoundResult = isTurnStage ? roundResults[currentRound - 2] : null;
  const allPlayers = buildAllPlayers(!isRoleSelection);

  return (
    <MockDataProvider mockData={mockData}>
      <div className="fixed inset-0 bg-gradient-to-b from-blue-400 to-blue-600 flex items-center justify-center p-2">
        <div className="w-full h-full flex items-center justify-center" style={{ maxWidth: '1400px' }}>
          {/* Battle Screen */}
          <div className="bg-white rounded-lg shadow-2xl border-4 border-gray-800 w-full h-full flex overflow-hidden relative">
            {/* Left Column - Game Interface */}
            <div className="flex-1 flex flex-col min-w-0">
              {/* Round Header */}
              <div className="bg-gray-800 text-white text-center flex-shrink-0 rounded-tl-lg flex items-center justify-center" style={{ height: '40px' }}>
                <h1 className="text-lg font-bold">
                  Tutorial 2 - Round {isRoleSelection ? 1 : currentRound - 1}/3
                </h1>
              </div>

              {/* Info Banners */}
              {isRoleSelection && (
                <div className="bg-yellow-50 border-b-4 border-yellow-400 px-4 py-3 flex-shrink-0">
                  <p className="text-sm text-yellow-900 font-semibold text-center">
                    Based on the action patterns in the Battle History, what role would complement the team?
                  </p>
                </div>
              )}

              {/* Battle Field */}
              <div className="flex-shrink-0" style={{ height: '35vh', minHeight: '250px', maxHeight: '400px' }}>
                <BattleField
                  enemyHealth={mockData.game.enemyHealth}
                  maxEnemyHealth={mockData.game.maxEnemyHealth}
                  teamHealth={mockData.game.teamHealth}
                  maxHealth={mockData.game.maxTeamHealth}
                  enemyIntent={currentRoundResult?.enemyIntent || null}
                  isRevealStage={!isRoleSelection}
                  showDamageAnimation={showDamageAnimation}
                  damageToEnemy={currentRoundResult?.damageToEnemy || 0}
                  damageToTeam={currentRoundResult?.damageToTeam || 0}
                  healAmount={currentRoundResult?.healAmount || 0}
                  actions={currentRoundResult?.actions || []}
                  allPlayers={allPlayers}
                  currentPlayerId="tutorial-player"
                  previousEnemyHealth={currentRoundResult?.previousEnemyHealth || mockData.game.enemyHealth}
                  previousTeamHealth={currentRoundResult?.previousTeamHealth || mockData.game.teamHealth}
                />
              </div>

              {/* Role Selection or Turn Results */}
              <div className="bg-white border-t-4 border-gray-700 flex-1 min-h-0 flex flex-col">
                <div className="flex-1 p-4 flex items-center justify-center overflow-auto">
                  {/* Role Selection */}
                  {isRoleSelection && (
                    <div className="w-full max-w-4xl">
                      <ActionMenu
                        selectedRole={selectedRole}
                        onRoleSelect={handleRoleSelect}
                        onSubmit={handleSubmit}
                        isRoleCommitted={false}
                        currentRole={null}
                        roundsRemaining={0}
                        submitted={false}
                        roleOrder={[ROLES.FIGHTER, ROLES.TANK, ROLES.HEALER]}
                      />
                    </div>
                  )}

                  {/* Turn Results - Rounds 2 and 3 */}
                  {isTurnStage && (
                    <div className="w-full">
                      <ResultsPanel
                        roundNumber={currentRoundResult.roundNum}
                        turnNumber={1}
                        actions={currentRoundResult.actions}
                        allPlayers={allPlayers}
                        currentPlayerId="tutorial-player"
                        enemyIntent={currentRoundResult.enemyIntent}
                        countdown={null}
                      />

                      <div className="mt-4 text-center">
                        <button
                          onClick={handleNextRound}
                          className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg text-lg shadow-lg transition-colors"
                        >
                          {currentRound < roundResults.length + 1 ? `Continue to Round ${currentRound - 1}` : "See Results"}
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>

            {/* Right Column - Battle History */}
            <div className="bg-gray-50 border-l-4 border-gray-700 overflow-hidden flex flex-col" style={{ width: '22%', minWidth: '280px', maxWidth: '350px' }}>
              <div className="bg-gray-800 text-white text-center flex-shrink-0 rounded-tr-lg flex items-center justify-center" style={{ height: '40px' }}>
                <h3 className="text-sm font-bold">📜 Battle History</h3>
              </div>
              <div className="flex-1 overflow-auto p-3 bg-white">
                <ActionHistory />
              </div>
            </div>

            {/* Outcome Overlay */}
            {showOutcome && outcome && (
              <div className="absolute inset-0 bg-black bg-opacity-60 flex items-center justify-center z-50">
                <div className="bg-blue-50 border-blue-400 border-4 rounded-xl p-8 max-w-2xl w-full shadow-2xl mx-4">
                  {/* Title */}
                  <div className="text-center mb-6">
                    <div className="text-8xl mb-4">🎮</div>
                    <h1 className="text-5xl font-bold text-blue-700 mb-2">
                      Tutorial Complete
                    </h1>
                    <p className="text-xl text-gray-700">{outcome.message}</p>
                  </div>

                  {/* Final Stats */}
                  <div className="bg-white rounded-lg p-6 mb-6 border-2 border-gray-300">
                    <h3 className="text-lg font-bold text-gray-800 mb-4 text-center">Final Battle Statistics</h3>
                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div className="text-center">
                        <div className="text-sm text-gray-600 mb-2">Enemy Health</div>
                        <div className="flex items-center justify-center gap-2">
                          <div className="text-3xl">👹</div>
                          <div className={`text-2xl font-bold ${outcome.enemyHealth === 0 ? 'text-gray-400 line-through' : 'text-red-600'}`}>
                            {outcome.enemyHealth} / 10
                          </div>
                        </div>
                      </div>
                      <div className="text-center">
                        <div className="text-sm text-gray-600 mb-2">Team Health</div>
                        <div className="flex items-center justify-center gap-2">
                          <div className="text-3xl">❤️</div>
                          <div className={`text-2xl font-bold ${outcome.teamHealth === 0 ? 'text-gray-400 line-through' : 'text-green-600'}`}>
                            {outcome.teamHealth} / 10
                          </div>
                        </div>
                      </div>
                    </div>

                    {/* Reveal actual roles */}
                    <div className="pt-4 border-t border-gray-300">
                      <p className="text-sm text-gray-700 mb-3 text-center font-semibold">
                        Actual Roles (in the real game, you'll need to infer these from action patterns):
                      </p>
                      <div className="flex gap-3 justify-center">
                        <div className="text-center bg-gray-100 rounded p-2">
                          <div className="text-2xl mb-1">🛡️</div>
                          <div className="text-xs text-gray-600">P1: Tank</div>
                          <div className="text-xs text-gray-500">(Defends when attacked)</div>
                        </div>
                        <div className="text-center bg-gray-100 rounded p-2">
                          <div className="text-2xl mb-1">💚</div>
                          <div className="text-xs text-gray-600">P2: Healer</div>
                          <div className="text-xs text-gray-500">(Heals when damaged)</div>
                        </div>
                        <div className="text-center bg-blue-100 border-2 border-blue-400 rounded p-2">
                          <div className="text-2xl mb-1">{["⚔️", "🛡️", "💚"][selectedRole]}</div>
                          <div className="text-xs text-gray-600">You: {ROLE_NAMES[selectedRole]}</div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Action Buttons */}
                  <div className="flex justify-center gap-4">
                    <button
                      onClick={handlePlayAgain}
                      className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg text-lg shadow-lg transition-colors"
                    >
                      Play Again
                    </button>
                    <button
                      onClick={handleStartMainGame}
                      className="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-8 rounded-lg text-lg shadow-lg transition-colors"
                    >
                      Start Main Game →
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </MockDataProvider>
  );
}

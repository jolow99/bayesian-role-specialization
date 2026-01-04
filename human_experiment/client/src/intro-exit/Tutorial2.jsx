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
  const [currentRound, setCurrentRound] = useState(1); // 1 = observing round 1, 2 = role selection, 3-4 = rounds 2-3
  const [roundResults, setRoundResults] = useState([]);
  const [mockData, setMockData] = useState(null);
  const [showOutcome, setShowOutcome] = useState(false);
  const [outcome, setOutcome] = useState(null);
  const [showDamageAnimation, setShowDamageAnimation] = useState(false);

  // Bot players: One Fighter (always attacks), One Tank (defends when enemy attacks)
  const actualBotRoles = [ROLES.FIGHTER, ROLES.TANK];

  useEffect(() => {
    // Initialize with round 1 observation - enemy attacks, so Tank will defend
    const round1Result = simulateRound(1, null, 10, 10, true); // true = enemy attacks, null = player hasn't chosen
    setRoundResults([round1Result]);

    const round1MockData = createMockDataForRound(round1Result, 1, false);
    setMockData(round1MockData);
  }, []);

  const getBotAction = (role, enemyAttacks) => {
    if (role === ROLES.FIGHTER) return "ATTACK";
    if (role === ROLES.TANK) return enemyAttacks ? "DEFEND" : "ATTACK";
    if (role === ROLES.HEALER) return "HEAL";
    return "ATTACK";
  };

  // Simulate a single round of combat
  const simulateRound = (roundNum, playerRole, currentEnemyHP, currentTeamHP, fixedEnemyAttacks = null) => {
    const enemyAttacks = fixedEnemyAttacks !== null ? fixedEnemyAttacks : Math.random() > 0.5;
    const enemyIntent = enemyAttacks ? "WILL_ATTACK" : "WILL_REST";

    const STR = 0.33, DEF = 0.33, SUP = 0.33;

    const bot1Action = getBotAction(actualBotRoles[0], enemyAttacks);
    const bot2Action = getBotAction(actualBotRoles[1], enemyAttacks);

    let playerAction, actions, roles;
    if (playerRole === null) {
      // Round 1, player hasn't chosen yet
      playerAction = null;
      actions = [bot1Action, bot2Action];
      roles = actualBotRoles;
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
    }

    let attackCount = actions.filter(a => a === "ATTACK").length;
    let damageToEnemy = attackCount * STR * 1.5;

    let hasDefender = actions.includes("DEFEND");
    let damageToTeam = 0;
    if (enemyAttacks) {
      const bossDamage = 2;
      if (hasDefender) {
        damageToTeam = Math.max(0, bossDamage - DEF * 3);
      } else {
        damageToTeam = bossDamage;
      }
    }

    let healCount = actions.filter(a => a === "HEAL").length;
    let healAmount = healCount * SUP * 2;

    const newEnemyHP = Math.max(0, currentEnemyHP - damageToEnemy);
    const newTeamHP = Math.max(0, Math.min(10, currentTeamHP - damageToTeam + healAmount));

    return {
      roundNum,
      enemyAttacks,
      enemyIntent,
      bot1Action,
      bot2Action,
      playerAction,
      playerRole,
      actions,
      roles,
      damageToEnemy: Math.round(damageToEnemy * 10) / 10,
      damageToTeam: Math.round(damageToTeam * 10) / 10,
      healAmount: Math.round(healAmount * 10) / 10,
      enemyHealth: Math.round(newEnemyHP * 10) / 10,
      teamHealth: Math.round(newTeamHP * 10) / 10,
      previousEnemyHealth: Math.round(currentEnemyHP * 10) / 10,
      previousTeamHealth: Math.round(currentTeamHP * 10) / 10
    };
  };

  const handleRoleSelect = (role) => {
    setSelectedRole(role);
  };

  const handleObservationContinue = () => {
    // Move to role selection
    setCurrentRound(2);
    const roleSelectionMockData = createMockDataForRoleSelection();
    setMockData(roleSelectionMockData);
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

    // Determine outcome
    let outcomeMessage, success;
    if (currentEnemyHP <= 0) {
      outcomeMessage = "Victory! Great job coordinating with your team!";
      success = true;
    } else if (currentTeamHP <= 0) {
      outcomeMessage = "Defeat! Your team was overwhelmed.";
      success = false;
    } else {
      if (selectedRole === ROLES.HEALER) {
        outcomeMessage = "Excellent! Your healing kept the team healthy while they attacked and defended.";
        success = true;
      } else if (selectedRole === ROLES.TANK) {
        outcomeMessage = "The team survived, but having two defenders is redundant. A healer would have been more helpful!";
        success = false;
      } else {
        outcomeMessage = "The team survived with heavy damage. Without healing, it was harder to stay healthy!";
        success = false;
      }
    }

    setOutcome({
      type: success ? "WIN" : "LOSE",
      message: outcomeMessage,
      success,
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

  const createMockDataForRoleSelection = () => {
    const round1Result = roundResults[0];
    const players = [
      { id: "bot-1", playerId: 0, stats: { STR: 2, DEF: 2, SUP: 2 } },
      { id: "bot-2", playerId: 1, stats: { STR: 2, DEF: 2, SUP: 2 } },
      { id: "tutorial-player", playerId: 2, stats: { STR: 2, DEF: 2, SUP: 2 } }
    ];

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
      teamHistory: []
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
      teamHistory: []
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
    setCurrentRound(1);
    setShowOutcome(false);
    setOutcome(null);

    // Re-initialize with round 1 observation
    const round1Result = simulateRound(1, null, 10, 10, true);
    setRoundResults([round1Result]);

    const round1MockData = createMockDataForRound(round1Result, 1, false);
    setMockData(round1MockData);
  };

  const handleStartMainGame = () => {
    next();
  };

  if (!mockData) return null;

  const isObservingRound1 = currentRound === 1;
  const isRoleSelection = currentRound === 2;
  const isTurnStage = currentRound >= 3;
  const currentRoundResult = isObservingRound1 ? roundResults[0] : (isTurnStage ? roundResults[currentRound - 2] : null);
  const allPlayers = buildAllPlayers(!isObservingRound1);

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
                  Tutorial 2 - Round {isRoleSelection ? 1 : (isObservingRound1 ? 1 : currentRound - 1)}/3
                </h1>
              </div>

              {/* Info Banners */}
              {isObservingRound1 && (
                <div className="bg-yellow-50 border-b-4 border-yellow-400 px-4 py-3 flex-shrink-0">
                  <p className="text-sm text-yellow-900 font-semibold text-center">
                    You'll observe what actions your teammates take. Based on their actions, what roles do you think they're playing?
                  </p>
                </div>
              )}

              {isRoleSelection && (
                <div className="bg-yellow-50 border-b-4 border-yellow-400 px-4 py-3 flex-shrink-0">
                  <p className="text-sm text-yellow-900 font-semibold text-center">
                    What role would complement the team best?
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
                  {/* Observation - Round 1 Results */}
                  {isObservingRound1 && (
                    <div className="w-full">
                      <ResultsPanel
                        roundNumber={1}
                        turnNumber={1}
                        actions={roundResults[0].actions}
                        allPlayers={allPlayers}
                        currentPlayerId="tutorial-player"
                        enemyIntent={roundResults[0].enemyIntent}
                        countdown={null}
                      />

                      <div className="mt-4 text-center">
                        <button
                          onClick={handleObservationContinue}
                          className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg text-lg shadow-lg transition-colors"
                        >
                          Choose Your Role
                        </button>
                      </div>
                    </div>
                  )}

                  {/* Role Selection */}
                  {isRoleSelection && (
                    <div className="w-full max-w-4xl">
                      {/* Reminder of what was observed */}
                      <div className="mb-6 bg-gray-50 rounded-lg p-4 border-2 border-gray-200">
                        <h4 className="font-semibold mb-3 text-center text-gray-700">What You Observed in Round 1:</h4>
                        <div className="flex gap-4 justify-center">
                          <div className="text-center bg-white border-2 border-gray-300 rounded p-3">
                            <div className="text-3xl mb-1">⚔️</div>
                            <div className="text-xs font-semibold text-gray-600">Player 1</div>
                            <div className="text-sm font-bold text-gray-700">Attacked</div>
                          </div>
                          <div className="text-center bg-white border-2 border-gray-300 rounded p-3">
                            <div className="text-3xl mb-1">🛡️</div>
                            <div className="text-xs font-semibold text-gray-600">Player 2</div>
                            <div className="text-sm font-bold text-gray-700">Defended</div>
                          </div>
                        </div>
                      </div>

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
                <div className={`${outcome.success ? 'bg-green-50 border-green-400' : 'bg-orange-50 border-orange-400'} border-4 rounded-xl p-8 max-w-2xl w-full shadow-2xl mx-4`}>
                  {/* Icon and Title */}
                  <div className="text-center mb-6">
                    <div className="text-8xl mb-4">{outcome.success ? '🎉' : '⚠️'}</div>
                    <h1 className={`text-5xl font-bold ${outcome.success ? 'text-green-700' : 'text-orange-700'} mb-2`}>
                      {outcome.success ? 'Success!' : 'Complete'}
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
                        In the real game, you'll need to infer roles from actions. Did you guess correctly?
                      </p>
                      <div className="flex gap-3 justify-center">
                        <div className="text-center bg-gray-100 rounded p-2">
                          <div className="text-2xl mb-1">⚔️</div>
                          <div className="text-xs text-gray-600">P1: Fighter</div>
                          <div className="text-xs text-gray-500">(Attacks)</div>
                        </div>
                        <div className="text-center bg-gray-100 rounded p-2">
                          <div className="text-2xl mb-1">🛡️</div>
                          <div className="text-xs text-gray-600">P2: Tank</div>
                          <div className="text-xs text-gray-500">(Defends)</div>
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

import React, { useState, useEffect } from "react";
import { MockDataProvider, TutorialWrapper } from "../components/tutorial";
import { BattleField } from "../components/BattleField";
import { ActionMenu } from "../components/ActionMenu";
import { ResultsPanel } from "../components/ResultsPanel";
import { ActionHistory } from "../components/ActionHistory";

const ROLES = { FIGHTER: 0, TANK: 1, HEALER: 2 };
const ROLE_NAMES = ["Fighter", "Tank", "Healer"];

export function Tutorial2({ next }) {
  const [selectedRole, setSelectedRole] = useState(null);
  const [mockData, setMockData] = useState(null);
  const [showOutcome, setShowOutcome] = useState(false);
  const [outcome, setOutcome] = useState(null);
  const [showDamageAnimation, setShowDamageAnimation] = useState(false);
  const [round1Turn1Result, setRound1Turn1Result] = useState(null);
  const [round1Turn2Result, setRound1Turn2Result] = useState(null);
  const [round2Turn1Result, setRound2Turn1Result] = useState(null);
  const [round2Turn2Result, setRound2Turn2Result] = useState(null);
  const [currentGameState, setCurrentGameState] = useState("initial"); // initial, round1-complete, role-selection, round2-turn1, round2-turn2, outcome
  const [tutorialComplete, setTutorialComplete] = useState(false);

  // Bot players: One Tank (defends when enemy attacks), One Healer (heals when health < 50%)
  const actualBotRoles = [ROLES.TANK, ROLES.HEALER];

  useEffect(() => {
    // Auto-play Round 1 on mount
    playRound1();
  }, []);

  const getBotAction = (role, enemyAttacks, teamHealth) => {
    if (role === ROLES.FIGHTER) return "ATTACK";
    if (role === ROLES.TANK) return enemyAttacks ? "DEFEND" : "ATTACK";
    if (role === ROLES.HEALER) {
      // Healer heals when team health <= 50% (which is 5 out of 10)
      return teamHealth <= 5 ? "HEAL" : "ATTACK";
    }
    return "ATTACK";
  };

  // Simulate a single turn of combat using real game stats
  const simulateTurn = (turnNum, playerRole, currentEnemyHP, currentTeamHP, enemyAttacks) => {
    const enemyIntent = enemyAttacks ? "WILL_ATTACK" : "WILL_REST";
    const STR = 2, DEF = 2, SUP = 2; // Real game stats
    const bossDamage = 7; // Tutorial 2 boss damage
    const maxTeamHealth = 10;

    // Calculate bot actions
    const bot1Action = getBotAction(actualBotRoles[0], enemyAttacks, currentTeamHP);
    const bot2Action = getBotAction(actualBotRoles[1], enemyAttacks, currentTeamHP);

    let playerAction, actions, roles, stats;
    if (playerRole === null) {
      // Player hasn't chosen yet - only 2 bots
      playerAction = null;
      actions = [bot1Action, bot2Action];
      roles = actualBotRoles;
      stats = [{ STR, DEF, SUP }, { STR, DEF, SUP }];
    } else {
      // Player has chosen - calculate their action
      playerAction = getBotAction(playerRole, enemyAttacks, currentTeamHP);
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

  const playRound1 = () => {
    // Round 1 Turn 1: Enemy attacks, starting from 10:10
    const r1t1 = simulateTurn(1, null, 10, 10, true);
    setRound1Turn1Result(r1t1);

    // Round 1 Turn 2: Enemy rests
    const r1t2 = simulateTurn(2, null, r1t1.enemyHealth, r1t1.teamHealth, false);
    setRound1Turn2Result(r1t2);

    // Update mock data to show Round 1 complete state
    const newMockData = createMockDataForRound1Complete(r1t1, r1t2);
    setMockData(newMockData);
    setCurrentGameState("round1-complete");
  };

  const createMockDataForRound1Complete = (turn1Result, turn2Result) => {
    const players = [
      { id: "bot-1", playerId: 0, stats: { STR: 2, DEF: 2, SUP: 2 } },
      { id: "bot-2", playerId: 1, stats: { STR: 2, DEF: 2, SUP: 2 } }
    ];

    const teamHistory = [
      {
        round: 1,
        turn: 1,
        enemyHealth: turn1Result.enemyHealth,
        teamHealth: turn1Result.teamHealth,
        enemyIntent: turn1Result.enemyIntent,
        actions: turn1Result.actions.map((action, idx) => ({
          playerId: idx,
          action: action
        }))
      },
      {
        round: 1,
        turn: 2,
        enemyHealth: turn2Result.enemyHealth,
        teamHealth: turn2Result.teamHealth,
        enemyIntent: turn2Result.enemyIntent,
        actions: turn2Result.actions.map((action, idx) => ({
          playerId: idx,
          action: action
        }))
      }
    ];

    return {
      game: {
        enemyHealth: turn2Result.enemyHealth,
        maxEnemyHealth: 10,
        teamHealth: turn2Result.teamHealth,
        maxTeamHealth: 10,
        treatment: {
          totalPlayers: 2,
          maxRounds: 2,
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
        roundNumber: 1
      },
      stage: {
        name: "turn2",
        stageType: "turn",
        turnNumber: 2
      },
      teamHistory: teamHistory
    };
  };

  const createMockDataForRoleSelection = () => {
    const players = [
      { id: "bot-1", playerId: 0, stats: { STR: 2, DEF: 2, SUP: 2 } },
      { id: "bot-2", playerId: 1, stats: { STR: 2, DEF: 2, SUP: 2 } },
      { id: "tutorial-player", playerId: 2, stats: { STR: 2, DEF: 2, SUP: 2 } }
    ];

    const teamHistory = [
      {
        round: 1,
        turn: 1,
        enemyHealth: round1Turn1Result.enemyHealth,
        teamHealth: round1Turn1Result.teamHealth,
        enemyIntent: round1Turn1Result.enemyIntent,
        actions: round1Turn1Result.actions.map((action, idx) => ({
          playerId: idx,
          action: action
        }))
      },
      {
        round: 1,
        turn: 2,
        enemyHealth: round1Turn2Result.enemyHealth,
        teamHealth: round1Turn2Result.teamHealth,
        enemyIntent: round1Turn2Result.enemyIntent,
        actions: round1Turn2Result.actions.map((action, idx) => ({
          playerId: idx,
          action: action
        }))
      }
    ];

    return {
      game: {
        enemyHealth: round1Turn2Result.enemyHealth,
        maxEnemyHealth: 10,
        teamHealth: round1Turn2Result.teamHealth,
        maxTeamHealth: 10,
        treatment: {
          totalPlayers: 3,
          maxRounds: 2,
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

  const createMockDataForRound2 = (turn1Result, turn2Result, currentTurn) => {
    const players = [
      { id: "bot-1", playerId: 0, stats: { STR: 2, DEF: 2, SUP: 2 } },
      { id: "bot-2", playerId: 1, stats: { STR: 2, DEF: 2, SUP: 2 } },
      { id: "tutorial-player", playerId: 2, stats: { STR: 2, DEF: 2, SUP: 2 } }
    ];

    const teamHistory = [
      {
        round: 1,
        turn: 1,
        enemyHealth: round1Turn1Result.enemyHealth,
        teamHealth: round1Turn1Result.teamHealth,
        enemyIntent: round1Turn1Result.enemyIntent,
        actions: round1Turn1Result.actions.map((action, idx) => ({
          playerId: idx,
          action: action
        }))
      },
      {
        round: 1,
        turn: 2,
        enemyHealth: round1Turn2Result.enemyHealth,
        teamHealth: round1Turn2Result.teamHealth,
        enemyIntent: round1Turn2Result.enemyIntent,
        actions: round1Turn2Result.actions.map((action, idx) => ({
          playerId: idx,
          action: action
        }))
      },
      {
        round: 2,
        turn: 1,
        enemyHealth: turn1Result.enemyHealth,
        teamHealth: turn1Result.teamHealth,
        enemyIntent: turn1Result.enemyIntent,
        actions: turn1Result.actions.map((action, idx) => ({
          playerId: idx,
          action: action
        }))
      }
    ];

    if (turn2Result) {
      teamHistory.push({
        round: 2,
        turn: 2,
        enemyHealth: turn2Result.enemyHealth,
        teamHealth: turn2Result.teamHealth,
        enemyIntent: turn2Result.enemyIntent,
        actions: turn2Result.actions.map((action, idx) => ({
          playerId: idx,
          action: action
        }))
      });
    }

    return {
      game: {
        enemyHealth: currentTurn.enemyHealth,
        maxEnemyHealth: 10,
        teamHealth: currentTurn.teamHealth,
        maxTeamHealth: 10,
        treatment: {
          totalPlayers: 3,
          maxRounds: 2,
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
        name: `turn${currentTurn.turnNum}`,
        stageType: "turn",
        turnNumber: currentTurn.turnNum
      },
      teamHistory: teamHistory
    };
  };

  const buildAllPlayers = () => {
    if (!mockData) return [];
    return mockData.players.map((p, idx) => ({
      type: (currentGameState === "role-selection" || currentGameState.startsWith("round2")) && idx === 2 ? "real" : "virtual",
      player: p,
      playerId: p.playerId,
      bot: ((currentGameState === "role-selection" || currentGameState.startsWith("round2")) && idx === 2) ? null : { stats: p.stats, playerId: p.playerId }
    }));
  };

  const handleRoleSelect = (role) => {
    setSelectedRole(role);
  };

  const handleSubmit = () => {
    if (selectedRole === null) return;

    // Simulate Round 2 with 2 turns based on selected role
    let currentEnemyHP = round1Turn2Result.enemyHealth;
    let currentTeamHP = round1Turn2Result.teamHealth;

    // Round 2 Turn 1: Enemy attacks
    const r2t1 = simulateTurn(1, selectedRole, currentEnemyHP, currentTeamHP, true);
    setRound2Turn1Result(r2t1);

    // Round 2 Turn 2: Enemy rests
    const r2t2 = simulateTurn(2, selectedRole, r2t1.enemyHealth, r2t1.teamHealth, false);
    setRound2Turn2Result(r2t2);

    // Determine outcome
    let outcomeMessage;
    if (r2t2.enemyHealth <= 0) {
      outcomeMessage = "The enemy was defeated!";
    } else if (r2t2.teamHealth <= 0) {
      outcomeMessage = "The team was defeated.";
    } else {
      outcomeMessage = "The battle concluded after 2 rounds.";
    }

    setOutcome({
      message: outcomeMessage,
      enemyHealth: r2t2.enemyHealth,
      teamHealth: r2t2.teamHealth
    });

    // Show round 2 turn 1
    const newMockData = createMockDataForRound2(r2t1, null, r2t1);
    setMockData(newMockData);
    setCurrentGameState("round2-turn1");
    setShowDamageAnimation(true);
    setTimeout(() => setShowDamageAnimation(false), 3000);
  };

  const handleNextToRound2Turn2 = () => {
    const newMockData = createMockDataForRound2(round2Turn1Result, round2Turn2Result, round2Turn2Result);
    setMockData(newMockData);
    setCurrentGameState("round2-turn2");
    setShowDamageAnimation(true);
    setTimeout(() => setShowDamageAnimation(false), 3000);
  };

  const handleShowOutcome = () => {
    setShowOutcome(true);
  };

  const handlePlayAgain = () => {
    setSelectedRole(null);
    setCurrentGameState("initial");
    setShowOutcome(false);
    setOutcome(null);
    setRound1Turn1Result(null);
    setRound1Turn2Result(null);
    setRound2Turn1Result(null);
    setRound2Turn2Result(null);
    setTutorialComplete(false);
    playRound1();
  };

  const handleStartMainGame = () => {
    next();
  };

  const handleProceedToRoleSelection = () => {
    const newMockData = createMockDataForRoleSelection();
    setMockData(newMockData);
    setCurrentGameState("role-selection");
  };

  const handleTutorialComplete = () => {
    setTutorialComplete(true);
  };

  // Define tutorial steps
  const tutorialSteps = [
    {
      targetId: null,
      tooltipPosition: "center",
      content: (
        <div>
          <h4 className="text-lg font-bold text-gray-900 mb-2">Strategic Role Selection</h4>
          <p className="text-sm text-gray-700 mb-2">
            In this tutorial, you'll learn how to analyze battle patterns and choose the best role to complement your team.
          </p>
          <p className="text-sm text-gray-700">
            Round 1 has already been played out. Let's examine what happened to determine the best role for Round 2.
          </p>
        </div>
      )
    },
    {
      targetId: "battle-history-r1t1",
      tooltipPosition: "left",
      content: (
        <div>
          <h4 className="text-lg font-bold text-gray-900 mb-2">Round 1, Turn 1</h4>
          <p className="text-sm text-gray-700 mb-2">
            The enemy attacked, dealing 6 damage. Player 1 chose DEFEND (reducing damage by 2) and Player 2 chose ATTACK (dealing 2 damage).
          </p>
          <p className="text-sm text-gray-700 font-semibold">
            Result: Enemy 10→{round1Turn1Result?.enemyHealth}, Team 10→{round1Turn1Result?.teamHealth}
          </p>
        </div>
      )
    },
    {
      targetId: "battle-history-r1t2",
      tooltipPosition: "left",
      content: (
        <div>
          <h4 className="text-lg font-bold text-gray-900 mb-2">Round 1, Turn 2</h4>
          <p className="text-sm text-gray-700 mb-2">
            The enemy rested (no attack). Player 1 chose ATTACK and Player 2 chose HEAL (restoring 2 health).
          </p>
          <p className="text-sm text-gray-700 font-semibold">
            Result: Enemy {round1Turn1Result?.enemyHealth}→{round1Turn2Result?.enemyHealth}, Team {round1Turn1Result?.teamHealth}→{round1Turn2Result?.teamHealth}
          </p>
        </div>
      )
    },
    {
      targetId: "action-menu",
      tooltipPosition: "top",
      content: (
        <div>
          <h4 className="text-lg font-bold text-gray-900 mb-2">Choose Your Role</h4>
          <p className="text-sm text-gray-700 mb-2">
            Based on these action patterns, what roles might the players have?
          </p>
          <p className="text-sm text-gray-700 font-semibold">
            Choose a role that you think best complements the team for Round 2.
          </p>
        </div>
      )
    }
  ];

  if (!mockData) return null;

  const allPlayers = buildAllPlayers();
  const isRoleSelection = currentGameState === "role-selection";
  const isRound1Complete = currentGameState === "round1-complete";
  const isRound2Turn1 = currentGameState === "round2-turn1";
  const isRound2Turn2 = currentGameState === "round2-turn2";

  // Get current turn result for display
  let currentTurnResult = null;
  if (isRound2Turn1 && round2Turn1Result) {
    currentTurnResult = round2Turn1Result;
  } else if (isRound2Turn2 && round2Turn2Result) {
    currentTurnResult = round2Turn2Result;
  }

  const content = (
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
                    Tutorial 2 - Round {isRoleSelection || isRound2Turn1 || isRound2Turn2 ? "2" : "1"}/2
                  </h1>
                </div>

                {/* Battle Field */}
                <div className="flex-shrink-0" style={{ height: '35vh', minHeight: '250px', maxHeight: '400px' }}>
                  <BattleField
                    enemyHealth={mockData.game.enemyHealth}
                    maxEnemyHealth={mockData.game.maxEnemyHealth}
                    teamHealth={mockData.game.teamHealth}
                    maxHealth={mockData.game.maxTeamHealth}
                    enemyIntent={currentTurnResult?.enemyIntent || null}
                    isRevealStage={!isRoleSelection && currentTurnResult !== null}
                    showDamageAnimation={showDamageAnimation}
                    damageToEnemy={currentTurnResult?.damageToEnemy || 0}
                    damageToTeam={currentTurnResult?.damageToTeam || 0}
                    healAmount={currentTurnResult?.healAmount || 0}
                    actions={currentTurnResult?.actions || []}
                    allPlayers={allPlayers}
                    currentPlayerId="tutorial-player"
                    previousEnemyHealth={currentTurnResult?.previousEnemyHealth || mockData.game.enemyHealth}
                    previousTeamHealth={currentTurnResult?.previousTeamHealth || mockData.game.teamHealth}
                  />
                </div>

                {/* Role Selection or Turn Results */}
                <div className="bg-white border-t-4 border-gray-700 flex-1 min-h-0 flex flex-col">
                  <div className="flex-1 p-4 flex items-center justify-center overflow-auto">
                    {/* Round 1 Complete - Show button to proceed */}
                    {isRound1Complete && (
                      <div className="w-full max-w-3xl text-center">
                        <div className="mb-6">
                          <h2 className="text-3xl font-bold text-gray-900 mb-4">Round 1 Complete</h2>
                          <p className="text-lg text-gray-700 mb-4">
                            Round 1 has concluded. Review the Battle History to understand the action patterns.
                          </p>
                        </div>
                        <button
                          onClick={handleProceedToRoleSelection}
                          className="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-8 rounded-lg text-xl shadow-lg transition-colors"
                        >
                          Choose Your Role for Round 2 →
                        </button>
                      </div>
                    )}

                    {/* Role Selection */}
                    {isRoleSelection && (
                      <div className="w-full max-w-4xl">
                        <div data-tutorial-id="action-menu">
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
                      </div>
                    )}

                    {/* Round 2 Turn 1 Results */}
                    {isRound2Turn1 && (
                      <div className="w-full">
                        <ResultsPanel
                          roundNumber={2}
                          turnNumber={1}
                          actions={round2Turn1Result.actions}
                          allPlayers={allPlayers}
                          currentPlayerId="tutorial-player"
                          enemyIntent={round2Turn1Result.enemyIntent}
                          countdown={null}
                        />
                        <div className="mt-4 text-center">
                          <button
                            onClick={handleNextToRound2Turn2}
                            className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg text-lg shadow-lg transition-colors"
                          >
                            Continue to Turn 2 →
                          </button>
                        </div>
                      </div>
                    )}

                    {/* Round 2 Turn 2 Results */}
                    {isRound2Turn2 && (
                      <div className="w-full">
                        <ResultsPanel
                          roundNumber={2}
                          turnNumber={2}
                          actions={round2Turn2Result.actions}
                          allPlayers={allPlayers}
                          currentPlayerId="tutorial-player"
                          enemyIntent={round2Turn2Result.enemyIntent}
                          countdown={null}
                        />
                        <div className="mt-4 text-center">
                          <button
                            onClick={handleShowOutcome}
                            className="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-lg text-lg shadow-lg transition-colors"
                          >
                            See Results →
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
                          {selectedRole !== null && (
                            <div className="text-center bg-blue-100 border-2 border-blue-400 rounded p-2">
                              <div className="text-2xl mb-1">{["⚔️", "🛡️", "💚"][selectedRole]}</div>
                              <div className="text-xs text-gray-600">You: {ROLE_NAMES[selectedRole]}</div>
                            </div>
                          )}
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

  // Show tutorial during role selection if not yet completed
  if (isRoleSelection && !tutorialComplete) {
    return (
      <TutorialWrapper steps={tutorialSteps} onComplete={handleTutorialComplete}>
        {content}
      </TutorialWrapper>
    );
  }

  return content;
}

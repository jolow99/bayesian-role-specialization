import React, { useState, useEffect } from "react";
import { MockDataProvider, TutorialWrapper } from "../components/tutorial";
import { BattleField } from "../components/BattleField";
import { ActionMenu } from "../components/ActionMenu";
import { ResultsPanel } from "../components/ResultsPanel";
import { ActionHistory } from "../components/ActionHistory";

const ROLES = { FIGHTER: 0, TANK: 1, HEALER: 2 };

export function Tutorial1({ next }) {
  const [selectedRole, setSelectedRole] = useState(null);
  const [currentRound, setCurrentRound] = useState(0); // 0 = role selection, 1-2 = rounds
  const [roundResults, setRoundResults] = useState([]);
  const [mockData, setMockData] = useState(null);
  const [showOutcome, setShowOutcome] = useState(false);
  const [outcome, setOutcome] = useState(null);
  const [showDamageAnimation, setShowDamageAnimation] = useState(false);
  const [tutorialComplete, setTutorialComplete] = useState(false);

  // Initialize with role selection state
  useEffect(() => {
    const initialMockData = createMockDataForRoleSelection();
    setMockData(initialMockData);
  }, []);

  // Define tutorial steps
  const tutorialSteps = [
    {
      targetId: "teammate-roles",
      tooltipPosition: "right",
      content: (
        <div>
          <h4 className="text-lg font-bold text-gray-900 mb-2">Seeing Teammate Roles</h4>
          <p className="text-sm text-gray-700 mb-2">
            In this game, you get to see what roles the other players chose before you choose your own role.
          </p>
          <p className="text-sm text-gray-700 font-semibold">
            Note: In the main game, you will only get to see the actions and not the roles that your teammates chose.
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
          <p className="text-sm text-gray-700">
            Depending on what you think best helps the team, choose your role.
          </p>
        </div>
      )
    }
  ];

  const handleTutorialComplete = () => {
    setTutorialComplete(true);
  };

  // Bot players both chose Fighter (indices 0 and 1)
  const botRoles = [ROLES.FIGHTER, ROLES.FIGHTER];

  const handleRoleSelect = (role) => {
    setSelectedRole(role);
  };

  // Simulate a single round of combat
  const simulateRound = (roundNum, playerRole, currentEnemyHP, currentTeamHP) => {
    // Enemy always attacks in tutorial for predictable outcomes
    const enemyIntent = "WILL_ATTACK";

    // Calculate team actions based on roles
    const roles = [...botRoles, playerRole];

    // Calculate damage to enemy: sum of all attackers' STR (each player has STR=2)
    let attackers = [];
    let defenders = [];
    let healers = [];

    roles.forEach((role, idx) => {
      if (role === ROLES.FIGHTER) {
        attackers.push(idx);
      } else if (role === ROLES.TANK) {
        defenders.push(idx);
      } else if (role === ROLES.HEALER) {
        healers.push(idx);
      }
    });

    // Everyone attacks (fighters, and healers attack too when not healing)
    // Healers only heal if team health <= 50%
    const willHeal = healers.length > 0 && currentTeamHP <= 3; // 3 is 50% of 6

    // Damage to enemy: attackers + healers who don't heal this turn
    let damageToEnemy = attackers.length * 2;
    if (!willHeal && healers.length > 0) {
      damageToEnemy += healers.length * 2; // Healers attack if not healing
    }

    // Boss always does 3 damage
    const bossDamage = 3;

    // Damage to team calculation
    let damageToTeam = bossDamage;

    // If there's a defender, reduce damage by highest DEF (each has DEF=2)
    if (defenders.length > 0) {
      damageToTeam = Math.max(0, bossDamage - 2); // Highest DEF is 2, so 3 - 2 = 1
    }

    // Healing - sum of all healers' SUP (each has SUP=2), only if they're healing
    let healAmount = willHeal ? healers.length * 2 : 0;

    // Determine actions for display
    const actions = roles.map(role => {
      if (role === ROLES.FIGHTER) return "ATTACK";
      if (role === ROLES.TANK) return "DEFEND";
      if (role === ROLES.HEALER) return willHeal ? "HEAL" : "ATTACK";
      return "ATTACK";
    });

    // Update health
    const newEnemyHP = Math.max(0, currentEnemyHP - damageToEnemy);
    const newTeamHP = Math.max(0, currentTeamHP - damageToTeam + healAmount);

    return {
      roundNum,
      enemyIntent,
      playerRole,
      actions,
      roles,
      damageToEnemy,
      damageToTeam,
      healAmount,
      enemyHealth: newEnemyHP,
      teamHealth: newTeamHP,
      previousEnemyHealth: currentEnemyHP,
      previousTeamHealth: currentTeamHP
    };
  };

  const handleSubmit = () => {
    if (selectedRole === null) return;

    // Simulate 2 rounds
    const results = [];
    let currentEnemyHP = 8;
    let currentTeamHP = 6;

    for (let i = 1; i <= 2; i++) {
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
    // If both reach 0, team loses
    if (currentTeamHP <= 0) {
      outcomeMessage = "Defeat! The enemy overwhelmed your team.";
      success = false;
    } else if (currentEnemyHP <= 0) {
      outcomeMessage = "Victory! Your team defeated the enemy!";
      success = true;
    } else {
      outcomeMessage = "The battle continues...";
      success = false;
    }

    setOutcome({
      type: success ? "WIN" : "LOSE",
      message: outcomeMessage,
      success,
      enemyHealth: currentEnemyHP,
      teamHealth: currentTeamHP
    });

    // Show first round
    const firstRound = results[0];
    const firstRoundMockData = createMockDataForRound(firstRound, 1, [firstRound]);
    setMockData(firstRoundMockData);
    setCurrentRound(1);
  };

  // Trigger damage animation when advancing rounds
  useEffect(() => {
    if (currentRound > 0) {
      setShowDamageAnimation(true);
      const timer = setTimeout(() => setShowDamageAnimation(false), 3000);
      return () => clearTimeout(timer);
    }
  }, [currentRound]);

  const handleNextRound = () => {
    if (currentRound < roundResults.length) {
      const nextRoundResult = roundResults[currentRound];
      // Pass all results up to and including the next round for history
      const nextRoundMockData = createMockDataForRound(
        nextRoundResult,
        currentRound + 1,
        roundResults.slice(0, currentRound + 1)
      );
      setMockData(nextRoundMockData);
      setCurrentRound(currentRound + 1);
    } else {
      // Show outcome overlay
      setShowOutcome(true);
    }
  };

  const createMockDataForRoleSelection = () => {
    const players = [
      { id: "bot-1", playerId: 0, stats: { STR: 2, DEF: 2, SUP: 2 } },
      { id: "bot-2", playerId: 1, stats: { STR: 2, DEF: 2, SUP: 2 } },
      { id: "tutorial-player", playerId: 2, stats: { STR: 2, DEF: 2, SUP: 2 } }
    ];

    return {
      game: {
        enemyHealth: 8,
        maxEnemyHealth: 8,
        teamHealth: 6,
        maxTeamHealth: 6,
        treatment: {
          totalPlayers: 3,
          maxRounds: 2,
          maxEnemyHealth: 8,
          maxTeamHealth: 6
        }
      },
      player: {
        id: "tutorial-player",
        playerId: 2,
        stats: { STR: 2, DEF: 2, SUP: 2 },
        roleOrder: [ROLES.FIGHTER, ROLES.TANK, ROLES.HEALER],
        stage: {},
        round: {},
        get: (key) => {
          if (key === "actionHistory") return [];
          return undefined;
        }
      },
      players: players,
      round: {
        roundNumber: 1
      },
      stage: {
        name: "roleSelection",
        stageType: "roleSelection"
      },
      teamHistory: []
    };
  };

  const createMockDataForRound = (roundResult, roundNum, previousResults) => {
    const players = [
      { id: "bot-1", playerId: 0, stats: { STR: 2, DEF: 2, SUP: 2 } },
      { id: "bot-2", playerId: 1, stats: { STR: 2, DEF: 2, SUP: 2 } },
      { id: "tutorial-player", playerId: 2, stats: { STR: 2, DEF: 2, SUP: 2 } }
    ];

    // Build team history from all previous rounds
    const teamHistory = [];
    const playerActionHistory = [];

    previousResults.forEach((result, idx) => {
      const roundNumber = idx + 1;

      // Add player's role to action history
      playerActionHistory.push({
        round: roundNumber,
        role: result.playerRole
      });

      // Add team history entry for this round's turn
      teamHistory.push({
        round: roundNumber,
        turn: 1,
        enemyIntent: result.enemyIntent,
        actions: result.actions.map((action, playerIdx) => ({
          action: action,
          playerId: playerIdx
        })),
        enemyHealth: result.enemyHealth,
        teamHealth: result.teamHealth
      });
    });

    return {
      game: {
        enemyHealth: roundResult.enemyHealth,
        maxEnemyHealth: 8,
        teamHealth: roundResult.teamHealth,
        maxTeamHealth: 6,
        treatment: {
          totalPlayers: 3,
          maxRounds: 2,
          maxEnemyHealth: 8,
          maxTeamHealth: 6
        }
      },
      player: {
        id: "tutorial-player",
        playerId: 2,
        stats: { STR: 2, DEF: 2, SUP: 2 },
        roleOrder: [ROLES.FIGHTER, ROLES.TANK, ROLES.HEALER],
        stage: {},
        round: {},
        get: (key) => {
          if (key === "actionHistory") return playerActionHistory;
          return undefined;
        }
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

  const buildAllPlayers = () => {
    if (!mockData) return [];
    return mockData.players.map((p, idx) => ({
      type: idx === 2 ? "real" : "virtual",
      player: p,
      playerId: p.playerId,
      bot: idx === 2 ? null : { stats: p.stats, playerId: p.playerId }
    }));
  };

  const handlePlayAgain = () => {
    setSelectedRole(null);
    setCurrentRound(0);
    setRoundResults([]);
    setShowOutcome(false);
    setOutcome(null);
    const initialMockData = createMockDataForRoleSelection();
    setMockData(initialMockData);
  };

  const handleContinueToTutorial2 = () => {
    next();
  };

  if (!mockData) return null;

  const allPlayers = buildAllPlayers();
  const isRoleSelection = currentRound === 0;
  const isTurnStage = currentRound > 0;
  const currentRoundResult = currentRound > 0 ? roundResults[currentRound - 1] : null;

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
                <h1 className="text-lg font-bold">Tutorial 1 - Round {currentRound > 0 ? currentRound : 1}/2</h1>
              </div>


              {/* Battle Field */}
              <div className="flex-shrink-0" style={{ height: '35vh', minHeight: '250px', maxHeight: '400px' }}>
                <BattleField
                  enemyHealth={mockData.game.enemyHealth}
                  maxEnemyHealth={mockData.game.maxEnemyHealth}
                  teamHealth={mockData.game.teamHealth}
                  maxHealth={mockData.game.maxTeamHealth}
                  enemyIntent={currentRoundResult?.enemyIntent || null}
                  isRevealStage={isTurnStage}
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
                      {/* Show teammate roles */}
                      <div className="mb-6 bg-gray-50 rounded-lg p-4 border-2 border-gray-200" data-tutorial-id="teammate-roles">
                        <h4 className="font-semibold mb-3 text-center text-gray-700">Your Teammates' Roles:</h4>
                        <div className="flex gap-4 justify-center">
                          <div className="text-center bg-red-50 border-2 border-red-300 rounded p-3">
                            <div className="text-3xl mb-1">⚔️</div>
                            <div className="text-xs font-semibold text-gray-600">Player 1</div>
                            <div className="text-sm font-bold text-red-700">Fighter</div>
                          </div>
                          <div className="text-center bg-red-50 border-2 border-red-300 rounded p-3">
                            <div className="text-3xl mb-1">⚔️</div>
                            <div className="text-xs font-semibold text-gray-600">Player 2</div>
                            <div className="text-sm font-bold text-red-700">Fighter</div>
                          </div>
                        </div>
                      </div>

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

                  {/* Turn Results */}
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

                      {/* Continue button */}
                      <div className="mt-4 text-center">
                        <button
                          onClick={handleNextRound}
                          className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg text-lg shadow-lg transition-colors"
                        >
                          {currentRound < roundResults.length ? `Continue to Round ${currentRound + 1}` : "See Results"}
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
                <div className={`${outcome.success ? 'bg-green-50 border-green-400' : 'bg-red-50 border-red-400'} border-4 rounded-xl p-8 max-w-2xl w-full shadow-2xl mx-4`}>
                  {/* Icon and Title */}
                  <div className="text-center mb-6">
                    <div className="text-8xl mb-4">{outcome.success ? '🎉' : '💀'}</div>
                    <h1 className={`text-5xl font-bold ${outcome.success ? 'text-green-700' : 'text-red-700'} mb-2`}>
                      {outcome.success ? 'Victory!' : 'Defeat'}
                    </h1>
                    <p className="text-xl text-gray-700">{outcome.message}</p>
                  </div>

                  {/* Final Stats */}
                  <div className="bg-white rounded-lg p-6 mb-6 border-2 border-gray-300">
                    <h3 className="text-lg font-bold text-gray-800 mb-4 text-center">Final Battle Statistics</h3>
                    <div className="grid grid-cols-2 gap-4">
                      <div className="text-center">
                        <div className="text-sm text-gray-600 mb-2">Enemy Health</div>
                        <div className="flex items-center justify-center gap-2">
                          <div className="text-3xl">👹</div>
                          <div className={`text-2xl font-bold ${outcome.enemyHealth === 0 ? 'text-gray-400 line-through' : 'text-red-600'}`}>
                            {outcome.enemyHealth} / 8
                          </div>
                        </div>
                      </div>
                      <div className="text-center">
                        <div className="text-sm text-gray-600 mb-2">Team Health</div>
                        <div className="flex items-center justify-center gap-2">
                          <div className="text-3xl">❤️</div>
                          <div className={`text-2xl font-bold ${outcome.teamHealth === 0 ? 'text-gray-400 line-through' : 'text-green-600'}`}>
                            {outcome.teamHealth} / 6
                          </div>
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
                      onClick={handleContinueToTutorial2}
                      className={`${outcome.success ? 'bg-green-600 hover:bg-green-700' : 'bg-gray-600 hover:bg-gray-700'} text-white font-bold py-3 px-6 rounded-lg text-lg shadow-lg transition-colors`}
                    >
                      Continue to Tutorial 2 →
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

  // Only show tutorial on first load during role selection
  if (isRoleSelection && !tutorialComplete) {
    return (
      <TutorialWrapper steps={tutorialSteps} onComplete={handleTutorialComplete}>
        {content}
      </TutorialWrapper>
    );
  }

  return content;
}

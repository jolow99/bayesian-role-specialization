import React, { useState, useEffect } from "react";
import { MockDataProvider, TutorialWrapper } from "../components/tutorial";
import { BattleField } from "../components/BattleField";
import { ActionMenu } from "../components/ActionMenu";
import { ResultsPanel } from "../components/ResultsPanel";
import { ActionHistory } from "../components/ActionHistory";
import { ROLES } from "../constants";

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
      targetId: "full-screen",
      tooltipPosition: "center",
      showBorder: false,
      content: (
        <div>
          <h3 className="text-lg font-bold text-gray-900 mb-3">Tutorial Game 1</h3>
          <p className="text-sm text-gray-700 mb-3">
            Let's play a practice game to see how everything works together!
          </p>
          <p className="text-sm text-gray-700">
            In this tutorial, you'll get to see your teammates' roles before choosing your own.
          </p>
        </div>
      )
    },
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

    // Calculate damage to enemy: sum of all fighters' STR (each player has STR=2)
    let fighters = [];
    let tanks = [];
    let medics = [];

    roles.forEach((role, idx) => {
      if (role === ROLES.FIGHTER) {
        fighters.push(idx);
      } else if (role === ROLES.TANK) {
        tanks.push(idx);
      } else if (role === ROLES.MEDIC) {
        medics.push(idx);
      }
    });

    // Everyone attacks (fighters, and medics attack too when not healing)
    // medics only heal if team health < 100%
    const willHeal = medics.length > 0 && currentTeamHP < 6; // max hp is 6

    // Damage to enemy: fighters + medics who don't heal this turn
    let damageToEnemy = fighters.length * 2;
    if (!willHeal && medics.length > 0) {
      damageToEnemy += medics.length * 2; // medics attack if not healing
    }

    // Boss always does 3 damage
    const bossDamage = 3;

    // Damage to team calculation
    let damageToTeam = bossDamage;

    // If there's a tank, reduce damage by highest DEF (each has DEF=2)
    if (tanks.length > 0) {
      damageToTeam = Math.max(0, bossDamage - 2); // Highest DEF is 2, so 3 - 2 = 1
    }

    // Healing - sum of all medics' SUP (each has SUP=2), only if they're healing
    let healAmount = willHeal ? medics.length * 2 : 0;

    // Determine actions for display
    const actions = roles.map(role => {
      if (role === ROLES.FIGHTER) return "ATTACK";
      if (role === ROLES.TANK) return "BLOCK";
      if (role === ROLES.MEDIC) return willHeal ? "HEAL" : "ATTACK";
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
    const firstRoundMockData = createMockDataForRound(firstRound, [firstRound]);
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

    const roundData = {
      roundNumber: 1,
      stageNumber: 0
    };

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
        roleOrder: [ROLES.FIGHTER, ROLES.TANK, ROLES.MEDIC],
        stage: {},
        round: {},
        get: (key) => {
          if (key === "actionHistory") return [];
          return undefined;
        }
      },
      players: players,
      round: {
        ...roundData,
        get: (key) => roundData[key]
      },
      stage: {
        name: "roleSelection",
        stageType: "roleSelection"
      }
    };
  };

  const createMockDataForRound = (roundResult, previousResults) => {
    const players = [
      { id: "bot-1", playerId: 0, stats: { STR: 2, DEF: 2, SUP: 2 } },
      { id: "bot-2", playerId: 1, stats: { STR: 2, DEF: 2, SUP: 2 } },
      { id: "tutorial-player", playerId: 2, stats: { STR: 2, DEF: 2, SUP: 2 } }
    ];

    // Build stage turns data from all previous results
    // In Tutorial1, each round is treated as a stage with 1 turn
    const playerActionHistory = [];
    const roundData = {
      roundNumber: 1,
      stageNumber: previousResults.length, // Track how many stages completed
    };

    previousResults.forEach((result, idx) => {
      const stageNumber = idx + 1;

      // Add player's role to action history (using stage instead of round)
      playerActionHistory.push({
        stage: stageNumber,
        role: result.playerRole
      });

      // Store turns for this stage in the new format
      roundData[`stage${stageNumber}Turns`] = [{
        turnNumber: 1,
        enemyIntent: result.enemyIntent,
        actions: result.actions,
        roles: result.roles,
        damageToEnemy: result.damageToEnemy,
        damageToTeam: result.damageToTeam,
        healAmount: result.healAmount,
        previousEnemyHealth: result.previousEnemyHealth,
        previousTeamHealth: result.previousTeamHealth,
        newEnemyHealth: result.enemyHealth,
        newTeamHealth: result.teamHealth
      }];
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
        roleOrder: [ROLES.FIGHTER, ROLES.TANK, ROLES.MEDIC],
        stage: {},
        round: {},
        get: (key) => {
          if (key === "actionHistory") return playerActionHistory;
          return undefined;
        }
      },
      players: players,
      round: {
        ...roundData,
        get: (key) => roundData[key]
      },
      stage: {
        name: `turn1`,
        stageType: "turn",
        turnNumber: 1
      }
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
      <div className="fixed inset-0 bg-gradient-to-b from-blue-400 to-blue-600 flex items-center justify-center p-2" data-tutorial-id="full-screen">
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
                  currentPlayerGameId={2}
                  previousEnemyHealth={currentRoundResult?.previousEnemyHealth || mockData.game.enemyHealth}
                  previousTeamHealth={currentRoundResult?.previousTeamHealth || mockData.game.teamHealth}
                  bossDamage={3}
                  enemyAttackProbability={1.0}
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
                            <div className="text-3xl mb-1">🤺</div>
                            <div className="text-xs font-semibold text-gray-600">Player 1</div>
                            <div className="text-sm font-bold text-red-700">Fighter</div>
                          </div>
                          <div className="text-center bg-red-50 border-2 border-red-300 rounded p-3">
                            <div className="text-3xl mb-1">🤺</div>
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
                          roleOrder={[ROLES.FIGHTER, ROLES.TANK, ROLES.MEDIC]}
                        />
                      </div>
                    </div>
                  )}

                  {/* Turn Results */}
                  {isTurnStage && (
                    <div className="w-full">
                      <ResultsPanel
                        stageNumber={currentRound}
                        turnNumber={1}
                        actions={currentRoundResult.actions}
                        allPlayers={allPlayers}
                        currentPlayerGameId={2}
                        enemyIntent={currentRoundResult.enemyIntent}
                        previousTeamHealth={currentRoundResult.previousTeamHealth}
                        newTeamHealth={currentRoundResult.teamHealth}
                        previousEnemyHealth={currentRoundResult.previousEnemyHealth}
                        newEnemyHealth={currentRoundResult.enemyHealth}
                        damageToTeam={currentRoundResult.damageToTeam}
                        damageToEnemy={currentRoundResult.damageToEnemy}
                        healAmount={currentRoundResult.healAmount}
                        onNextTurn={handleNextRound}
                        nextButtonLabel={currentRound < roundResults.length ? `Continue to Round ${currentRound + 1}` : "See Results"}
                      />
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

                  {/* Explanation based on role choice */}
                  <div className="bg-white rounded-lg p-6 mb-6 border-2 border-gray-300">
                    <h3 className="text-lg font-bold text-gray-800 mb-4 text-center">What Happened</h3>

                    {selectedRole === ROLES.FIGHTER && (
                      <div className="text-gray-700 space-y-3">
                        <p>
                          <span className="font-semibold">You chose Fighter 🤺</span> — With 3 Fighters attacking (2 damage each), your team dealt 6 damage per turn.
                        </p>
                        <p>
                          However, with no one to block or heal, the boss's attacks (3 damage per turn) went unmitigated:
                        </p>
                        <ul className="list-disc list-inside ml-2 space-y-1 text-sm">
                          <li>Turn 1: Team dealt 6 damage to boss (8 → 2 HP). Boss dealt 3 damage to team (6 → 3 HP)</li>
                          <li>Turn 2: Team dealt 6 damage to boss (2 → 0 HP). Boss dealt 3 damage to team (3 → 0 HP)</li>
                        </ul>
                        <p className="font-semibold text-red-600">
                          Both sides were defeated in the same turn. The team must survive to win!
                        </p>
                      </div>
                    )}

                    {selectedRole === ROLES.TANK && (
                      <div className="text-gray-700 space-y-3">
                        <p>
                          <span className="font-semibold">You chose Tank 🛡️</span> — Your 2 Fighter teammates dealt 4 damage per turn, while you blocked to protect the team.
                        </p>
                        <p>
                          The boss deals 3 damage per turn, but your DEF of 2 reduced it to only 1 damage:
                        </p>
                        <ul className="list-disc list-inside ml-2 space-y-1 text-sm">
                          <li>Turn 1: Team dealt 4 damage to boss (8 → 4 HP). Boss dealt 3 - 2 = 1 damage to team (6 → 5 HP)</li>
                          <li>Turn 2: Team dealt 4 damage to boss (4 → 0 HP). Boss dealt 3 - 2 = 1 damage to team (5 → 4 HP)</li>
                        </ul>
                        <p className="font-semibold text-green-600">
                          Your team survived with 4 HP remaining, securing the victory!
                        </p>
                      </div>
                    )}

                    {selectedRole === ROLES.MEDIC && (
                      <div className="text-gray-700 space-y-3">
                        <p>
                          <span className="font-semibold">You chose Medic 💚</span> — Your healing kept the team alive.
                        </p>
                        <p>
                          Medics only heal when the team is damaged. At full health, they act like a fighter instead:
                        </p>
                        <ul className="list-disc list-inside ml-2 space-y-1 text-sm">
                          <li>Turn 1: Team at full HP, so you attacked. Team dealt 6 damage to boss (8 → 2 HP). Boss dealt 3 damage to team (6 → 3 HP)</li>
                          <li>Turn 2: Team damaged, so you healed for 2. Team dealt 4 damage to boss (2 → 0 HP). Boss dealt 3 damage, you healed 2 (3 - 3 + 2 = 2 HP)</li>
                        </ul>
                        <p className="font-semibold text-green-600">
                          Your team survived with 2 HP remaining, securing the victory!
                        </p>
                      </div>
                    )}
                  </div>

                  {/* Final Stats */}
                  <div className="bg-gray-100 rounded-lg p-4 mb-6 border border-gray-300">
                    <div className="grid grid-cols-2 gap-4">
                      <div className="text-center">
                        <div className="text-sm text-gray-600 mb-1">Team Health</div>
                        <div className="flex items-center justify-center gap-2">
                          <div className="text-2xl">❤️</div>
                          <div className={`text-xl font-bold ${outcome.teamHealth === 0 ? 'text-gray-400 line-through' : 'text-green-600'}`}>
                            {outcome.teamHealth} / 6
                          </div>
                        </div>
                      </div>
                      <div className="text-center">
                        <div className="text-sm text-gray-600 mb-1">Boss Health</div>
                        <div className="flex items-center justify-center gap-2">
                          <div className="text-2xl">👹</div>
                          <div className={`text-xl font-bold ${outcome.enemyHealth === 0 ? 'text-gray-400 line-through' : 'text-red-600'}`}>
                            {outcome.enemyHealth} / 8
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
                      {outcome.success ? 'Play Again' : 'Try Again'}
                    </button>
                    {outcome.success && (
                      <button
                        onClick={handleContinueToTutorial2}
                        className="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-lg text-lg shadow-lg transition-colors"
                      >
                        Continue to Tutorial 2 →
                      </button>
                    )}
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

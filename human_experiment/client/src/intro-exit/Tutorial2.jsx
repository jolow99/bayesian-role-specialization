import React, { useState } from "react";
import { Button } from "../components/Button";
import { MockDataProvider } from "../components/tutorial";
import { BattleField } from "../components/BattleField";
import { ResultsPanel } from "../components/ResultsPanel";

const ROLES = { FIGHTER: 0, TANK: 1, HEALER: 2 };
const ROLE_NAMES = ["Fighter", "Tank", "Healer"];

export function Tutorial2({ next }) {
  const [selectedRole, setSelectedRole] = useState(null);
  const [currentScreen, setCurrentScreen] = useState("intro"); // intro, round1, selection, round2, round3, final
  const [roundResults, setRoundResults] = useState([]);
  const [mockData, setMockData] = useState(null);
  const [outcome, setOutcome] = useState(null);

  // Bot players: One Fighter (always attacks), One Tank (defends when enemy attacks)
  const actualBotRoles = [ROLES.FIGHTER, ROLES.TANK];

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

    let playerAction;
    if (playerRole === null) {
      playerAction = null; // Round 1, player hasn't chosen yet
    } else if (playerRole === ROLES.FIGHTER) {
      playerAction = "ATTACK";
    } else if (playerRole === ROLES.TANK) {
      playerAction = enemyAttacks ? "DEFEND" : "ATTACK";
    } else if (playerRole === ROLES.HEALER) {
      playerAction = currentTeamHP <= 5 ? "HEAL" : "ATTACK";
    } else {
      playerAction = "ATTACK";
    }

    const actions = playerRole === null
      ? [bot1Action, bot2Action]
      : [bot1Action, bot2Action, playerAction];

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
      damageToEnemy: Math.round(damageToEnemy * 10) / 10,
      damageToTeam: Math.round(damageToTeam * 10) / 10,
      healAmount: Math.round(healAmount * 10) / 10,
      enemyHealth: Math.round(newEnemyHP * 10) / 10,
      teamHealth: Math.round(newTeamHP * 10) / 10,
      previousEnemyHealth: Math.round(currentEnemyHP * 10) / 10,
      previousTeamHealth: Math.round(currentTeamHP * 10) / 10
    };
  };

  const handleStartTutorial = () => {
    // Simulate round 1 with enemy attacking (so Tank will defend)
    const round1Result = simulateRound(1, null, 10, 10, true); // true = enemy attacks, null = player hasn't chosen
    setRoundResults([round1Result]);

    // Create mock data for round 1
    const round1MockData = createMockDataForRound(round1Result, 1, false);
    setMockData(round1MockData);
    setCurrentScreen("round1");
  };

  const handleRoleSelect = (role) => {
    setSelectedRole(role);
  };

  const handleConfirm = () => {
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
    let outcomeMessage;
    let success = false;
    if (currentEnemyHP <= 0) {
      outcomeMessage = "Victory! Great job coordinating with your team!";
      success = true;
    } else if (currentTeamHP <= 0) {
      outcomeMessage = "Defeat! Your team was overwhelmed.";
    } else {
      if (selectedRole === ROLES.HEALER) {
        outcomeMessage = "Excellent! Your healing kept the team healthy while they attacked and defended.";
        success = true;
      } else if (selectedRole === ROLES.TANK) {
        outcomeMessage = "The team survived, but having two defenders is redundant. A healer would have been more helpful!";
      } else {
        outcomeMessage = "The team survived with heavy damage. Without healing, it was harder to stay healthy!";
      }
    }

    setOutcome({ message: outcomeMessage, success, enemyHealth: currentEnemyHP, teamHealth: currentTeamHP });

    // Create mock data for round 2
    const round2MockData = createMockDataForRound(results[1], 2, true);
    setMockData(round2MockData);
    setCurrentScreen("round2");
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
        stage: {},
        round: {}
      },
      players: players,
      round: {
        roundNumber: roundNum,
        [`turn1Intent`]: roundResult.enemyIntent,
        [`turn1Actions`]: roundResult.actions,
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

  const handleNextRound = () => {
    const currentRoundNum = parseInt(currentScreen.replace("round", ""));
    if (currentRoundNum < roundResults.length) {
      const nextRound = roundResults[currentRoundNum];
      const newMockData = createMockDataForRound(nextRound, currentRoundNum + 1, true);
      setMockData(newMockData);
      setCurrentScreen(`round${currentRoundNum + 1}`);
    } else {
      setCurrentScreen("final");
    }
  };

  const handleNext = () => {
    next();
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

  // Intro screen
  if (currentScreen === "intro") {
    return (
      <div className="p-8 max-w-4xl mx-auto">
        <h3 className="text-2xl font-bold text-gray-900 mb-4 text-center">
          Tutorial Round 2
        </h3>

        <div className="bg-yellow-50 border border-yellow-300 rounded-lg p-4 mb-6">
          <p className="text-sm text-yellow-900 mb-2">
            This time, you won't see what roles your teammates chose at the start.
          </p>
          <p className="text-sm text-yellow-900 font-semibold">
            You'll observe the first round, then choose your role based on what actions you saw.
            Your role will then be locked for the next 2 rounds.
          </p>
        </div>

        <div className="flex justify-center">
          <Button handleClick={handleStartTutorial} autoFocus>
            <p>Begin Tutorial Round 2</p>
          </Button>
        </div>
      </div>
    );
  }

  // Round 1 observation
  if (currentScreen === "round1") {
    const result = roundResults[0];
    const allPlayers = buildAllPlayers(false);

    return (
      <MockDataProvider mockData={mockData}>
        <div className="fixed inset-0 bg-gradient-to-b from-blue-400 to-blue-600 flex items-center justify-center p-2">
          <div className="w-full h-full flex items-center justify-center" style={{ maxWidth: '1400px' }}>
            <div className="bg-white rounded-lg shadow-2xl border-4 border-gray-800 w-full h-full flex flex-col overflow-hidden">
              <div className="bg-gray-800 text-white text-center py-3">
                <h1 className="text-xl font-bold">Tutorial Round 2 - Observing Round 1</h1>
              </div>

              <div className="flex-1 flex flex-col p-6 overflow-auto">
                <div className="bg-yellow-50 border border-yellow-300 rounded-lg p-4 mb-4">
                  <p className="text-sm text-yellow-900 mb-2">
                    Here's what happened in the first round. Pay attention to what actions your teammates took!
                  </p>
                  <p className="text-sm text-yellow-900 font-semibold">
                    Based on their actions, what roles do you think they're playing? What role would complement the team best?
                  </p>
                </div>

                <div className="mb-6" style={{ height: '300px' }}>
                  <BattleField
                    enemyHealth={mockData.game.enemyHealth}
                    maxEnemyHealth={mockData.game.maxEnemyHealth}
                    teamHealth={mockData.game.teamHealth}
                    maxHealth={mockData.game.maxTeamHealth}
                    enemyIntent={result.enemyIntent}
                    isRevealStage={true}
                    showDamageAnimation={true}
                    damageToEnemy={result.damageToEnemy}
                    damageToTeam={result.damageToTeam}
                    healAmount={result.healAmount}
                    actions={result.actions}
                    allPlayers={allPlayers}
                    currentPlayerId="tutorial-player"
                    previousEnemyHealth={result.previousEnemyHealth}
                    previousTeamHealth={result.previousTeamHealth}
                  />
                </div>

                <div className="flex-1">
                  <ResultsPanel
                    roundNumber={result.roundNum}
                    turnNumber={1}
                    actions={result.actions}
                    allPlayers={allPlayers}
                    currentPlayerId="tutorial-player"
                    enemyIntent={result.enemyIntent}
                    countdown={null}
                  />
                </div>
              </div>

              <div className="bg-gray-100 border-t-4 border-gray-700 p-4 flex justify-center">
                <Button handleClick={() => setCurrentScreen("selection")} autoFocus>
                  <p>Choose Your Role</p>
                </Button>
              </div>
            </div>
          </div>
        </div>
      </MockDataProvider>
    );
  }

  // Role selection screen
  if (currentScreen === "selection") {
    const round1Result = roundResults[0];

    return (
      <div className="p-8 max-w-4xl mx-auto">
        <h3 className="text-2xl font-bold text-gray-900 mb-4 text-center">
          Tutorial Round 2 - Choose Your Role
        </h3>

        <div className="bg-yellow-50 border border-yellow-300 rounded-lg p-4 mb-6">
          <p className="text-sm text-yellow-900 mb-2">
            Remember what you saw in Round 1:
          </p>
          <div className="flex gap-4 justify-center my-3">
            <div className="bg-white rounded border border-yellow-400 p-2 text-center text-xs">
              <div className="text-xl mb-1">⚔️</div>
              <div>P1: <strong>Attack</strong></div>
            </div>
            <div className="bg-white rounded border border-yellow-400 p-2 text-center text-xs">
              <div className="text-xl mb-1">🛡️</div>
              <div>P2: <strong>Defend</strong></div>
            </div>
          </div>
          <p className="text-sm text-yellow-900 font-semibold">
            Based on their actions, what role do you think would complement the team best?
          </p>
          <p className="text-sm text-yellow-900 mt-2 text-xs">
            (Your role will be locked for rounds 2 and 3)
          </p>
        </div>

        <div className="bg-white border-2 border-gray-200 rounded-lg p-6 mb-6">
          <h4 className="font-semibold mb-4 text-center">Choose Your Role:</h4>
          <div className="flex gap-4 justify-center">
            {[ROLES.FIGHTER, ROLES.TANK, ROLES.HEALER].map((role) => (
              <button
                key={role}
                onClick={() => handleRoleSelect(role)}
                className={`flex flex-col items-center p-4 rounded-lg border-2 transition-all
                  ${selectedRole === role
                    ? 'border-blue-500 bg-blue-50 shadow-lg'
                    : 'border-gray-300 bg-white hover:border-blue-300'}`}
              >
                <div className="text-4xl mb-2">{["⚔️", "🛡️", "💚"][role]}</div>
                <div className="text-lg font-semibold">{ROLE_NAMES[role]}</div>
                <div className="text-xs text-gray-500 mt-1">
                  {role === ROLES.FIGHTER && "Attack enemies"}
                  {role === ROLES.TANK && "Defend team"}
                  {role === ROLES.HEALER && "Heal team"}
                </div>
              </button>
            ))}
          </div>
        </div>

        <div className="flex justify-center">
          <Button
            handleClick={handleConfirm}
            disabled={selectedRole === null}
          >
            <p>Confirm Role & Continue</p>
          </Button>
        </div>
      </div>
    );
  }

  // Rounds 2 and 3 playback
  if (currentScreen.startsWith("round") && currentScreen !== "round1") {
    const roundNum = parseInt(currentScreen.replace("round", ""));
    const result = roundResults[roundNum - 1];
    const allPlayers = buildAllPlayers(true);

    return (
      <MockDataProvider mockData={mockData}>
        <div className="fixed inset-0 bg-gradient-to-b from-blue-400 to-blue-600 flex items-center justify-center p-2">
          <div className="w-full h-full flex items-center justify-center" style={{ maxWidth: '1400px' }}>
            <div className="bg-white rounded-lg shadow-2xl border-4 border-gray-800 w-full h-full flex flex-col overflow-hidden">
              <div className="bg-gray-800 text-white text-center py-3">
                <h1 className="text-xl font-bold">Tutorial - Round {result.roundNum} of 3</h1>
              </div>

              <div className="flex-1 flex flex-col p-6 overflow-auto">
                <div className="mb-6" style={{ height: '300px' }}>
                  <BattleField
                    enemyHealth={mockData.game.enemyHealth}
                    maxEnemyHealth={mockData.game.maxEnemyHealth}
                    teamHealth={mockData.game.teamHealth}
                    maxHealth={mockData.game.maxTeamHealth}
                    enemyIntent={result.enemyIntent}
                    isRevealStage={true}
                    showDamageAnimation={true}
                    damageToEnemy={result.damageToEnemy}
                    damageToTeam={result.damageToTeam}
                    healAmount={result.healAmount}
                    actions={result.actions}
                    allPlayers={allPlayers}
                    currentPlayerId="tutorial-player"
                    previousEnemyHealth={result.previousEnemyHealth}
                    previousTeamHealth={result.previousTeamHealth}
                  />
                </div>

                <div className="flex-1">
                  <ResultsPanel
                    roundNumber={result.roundNum}
                    turnNumber={1}
                    actions={result.actions}
                    allPlayers={allPlayers}
                    currentPlayerId="tutorial-player"
                    enemyIntent={result.enemyIntent}
                    countdown={null}
                  />
                </div>
              </div>

              <div className="bg-gray-100 border-t-4 border-gray-700 p-4 flex justify-center">
                <Button handleClick={handleNextRound} autoFocus>
                  <p>{roundNum < 3 ? `Continue to Round ${roundNum + 1}` : "See Final Results"}</p>
                </Button>
              </div>
            </div>
          </div>
        </div>
      </MockDataProvider>
    );
  }

  // Final summary
  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h3 className="text-2xl font-bold text-gray-900 mb-4 text-center">
        Tutorial Round 2 - Final Results
      </h3>

      <div className={`border-2 rounded-lg p-6 mb-6 ${outcome.success ? 'bg-green-50 border-green-300' : 'bg-orange-50 border-orange-300'}`}>
        <p className="text-lg font-semibold mb-2 text-center">
          {outcome.message}
        </p>
      </div>

      <div className="bg-gray-50 rounded-lg p-6 mb-6">
        <h4 className="font-semibold mb-3">Final State:</h4>
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="text-center">
            <div className="text-sm text-gray-600">Enemy Health</div>
            <div className="text-3xl font-bold text-red-600">{outcome.enemyHealth}</div>
          </div>
          <div className="text-center">
            <div className="text-sm text-gray-600">Team Health</div>
            <div className="text-3xl font-bold text-green-600">{outcome.teamHealth}</div>
          </div>
        </div>

        <div className="pt-4 border-t border-gray-300">
          <p className="text-sm text-gray-700 mb-2 text-center"><strong>The team roles were:</strong></p>
          <div className="flex gap-4 justify-center">
            <div className="text-center bg-gray-100 rounded p-3">
              <div className="text-3xl mb-1">⚔️</div>
              <div className="text-xs text-gray-600">P1: Fighter</div>
              <div className="text-xs text-gray-500">(Always attacks)</div>
            </div>
            <div className="text-center bg-gray-100 rounded p-3">
              <div className="text-3xl mb-1">🛡️</div>
              <div className="text-xs text-gray-600">P2: Tank</div>
              <div className="text-xs text-gray-500">(Defends when enemy attacks)</div>
            </div>
            <div className="text-center bg-blue-100 border-2 border-blue-400 rounded p-3">
              <div className="text-3xl mb-1">{["⚔️", "🛡️", "💚"][selectedRole]}</div>
              <div className="text-xs text-gray-600">You: {ROLE_NAMES[selectedRole]}</div>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded p-4 mb-6">
        <p className="text-sm text-blue-800 mb-2">
          <strong>Key Learning:</strong> In the real game, you won't see what roles your teammates chose.
          You'll need to infer their roles from the actions they take each round.
        </p>
        <p className="text-sm text-blue-800">
          For example, if you see someone consistently attacking, they're likely a Fighter.
          If someone defends when the enemy attacks, they're probably a Tank.
          Pay attention to patterns and coordinate your role accordingly!
        </p>
      </div>

      <div className="bg-green-50 border border-green-300 rounded p-4 mb-6">
        <p className="text-sm text-green-800 font-semibold">
          You're now ready for the main game! Remember to pay attention to what actions
          your teammates are taking and coordinate your role accordingly.
        </p>
      </div>

      <div className="flex justify-center">
        <Button handleClick={handleNext} autoFocus>
          <p>Start Main Game</p>
        </Button>
      </div>
    </div>
  );
}

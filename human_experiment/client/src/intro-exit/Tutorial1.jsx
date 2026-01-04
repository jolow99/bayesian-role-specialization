import React, { useState } from "react";
import { Button } from "../components/Button";
import { MockDataProvider } from "../components/tutorial";
import { BattleField } from "../components/BattleField";
import { ActionMenu } from "../components/ActionMenu";
import { ResultsPanel } from "../components/ResultsPanel";
import { GameEndScreen } from "../components/GameEndScreen";

const ROLES = { FIGHTER: 0, TANK: 1, HEALER: 2 };
const ROLE_NAMES = ["Fighter", "Tank", "Healer"];

export function Tutorial1({ next }) {
  const [selectedRole, setSelectedRole] = useState(null);
  const [currentScreen, setCurrentScreen] = useState("intro"); // intro, selection, round1, round2, final
  const [roundResults, setRoundResults] = useState([]);
  const [mockData, setMockData] = useState(null);
  const [outcome, setOutcome] = useState(null);

  // Bot players both chose Fighter (indices 0 and 1)
  const botRoles = [ROLES.FIGHTER, ROLES.FIGHTER];

  const handleStartTutorial = () => {
    setCurrentScreen("selection");
  };

  const handleRoleSelect = (role) => {
    setSelectedRole(role);
  };

  // Simulate a single round of combat
  const simulateRound = (roundNum, playerRole, currentEnemyHP, currentTeamHP) => {
    // Enemy always attacks in tutorial for predictable outcomes
    const enemyIntent = "WILL_ATTACK";

    // Calculate team actions based on roles
    const roles = [...botRoles, playerRole];

    // Count attacks - 2 Fighters deal 5 damage each, 3 Fighters deal 3.5 damage each
    let attackCount = roles.filter(r => r === ROLES.FIGHTER).length;
    let damageToEnemy;
    if (attackCount === 2) {
      damageToEnemy = 5; // 2 fighters = 5 damage per round
    } else if (attackCount === 3) {
      damageToEnemy = 3.5; // 3 fighters = 3.5 damage
    } else {
      damageToEnemy = 0;
    }

    // Check for defense
    let hasDefender = roles.includes(ROLES.TANK);
    let damageToTeam = 0;
    const bossDamage = 6; // Boss deals 6 damage per attack
    if (hasDefender) {
      damageToTeam = 1; // Tank reduces it to 1 damage
    } else {
      damageToTeam = bossDamage;
    }

    // Check for healing
    let hasHealer = roles.includes(ROLES.HEALER);
    let healAmount = 0;
    if (hasHealer && currentTeamHP < 10) {
      healAmount = 3; // Healer restores 3 HP
    }

    // Determine actions for display
    const actions = roles.map(role => {
      if (role === ROLES.FIGHTER) return "ATTACK";
      if (role === ROLES.TANK) return "DEFEND";
      if (role === ROLES.HEALER) return healAmount > 0 ? "HEAL" : "ATTACK";
      return "ATTACK";
    });

    // Update health
    const newEnemyHP = Math.max(0, currentEnemyHP - damageToEnemy);
    const newTeamHP = Math.max(0, Math.min(10, currentTeamHP - damageToTeam + healAmount));

    return {
      roundNum,
      enemyIntent,
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

  const handleConfirm = () => {
    // Simulate 2 rounds
    const results = [];
    let currentEnemyHP = 10;
    let currentTeamHP = 10;

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
    let outcomeMessage;
    let success = false;
    if (currentEnemyHP <= 0) {
      outcomeMessage = "Victory! By choosing a Tank or Healer role, you complemented the two Fighters perfectly!";
      success = true;
    } else if (currentTeamHP <= 0) {
      outcomeMessage = "Defeat! With three Fighters and no Tank or Healer, your team couldn't survive the enemy's attacks. Try a different role!";
    } else {
      outcomeMessage = "The battle continues... Consider choosing Tank or Healer to complement your Fighter teammates!";
    }

    setOutcome({ message: outcomeMessage, success, enemyHealth: currentEnemyHP, teamHealth: currentTeamHP });

    // Create initial mock data for first round
    const firstRound = results[0];
    const initialMockData = createMockDataForRound(firstRound, 1);
    setMockData(initialMockData);
    setCurrentScreen("round1");
  };

  const createMockDataForRound = (roundResult, roundNum) => {
    // Build allPlayers array
    const players = [
      { id: "bot-1", playerId: 0, stats: { STR: 2, DEF: 2, SUP: 2 } },
      { id: "bot-2", playerId: 1, stats: { STR: 2, DEF: 2, SUP: 2 } },
      { id: "tutorial-player", playerId: 2, stats: { STR: 2, DEF: 2, SUP: 2 } }
    ];

    return {
      game: {
        enemyHealth: roundResult.enemyHealth,
        maxEnemyHealth: 10,
        teamHealth: roundResult.teamHealth,
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

  const handleNextRound = () => {
    if (currentScreen === "round1" && roundResults.length > 1) {
      const secondRound = roundResults[1];
      const newMockData = createMockDataForRound(secondRound, 2);
      setMockData(newMockData);
      setCurrentScreen("round2");
    } else {
      setCurrentScreen("final");
    }
  };

  const handlePlayAgain = () => {
    setSelectedRole(null);
    setCurrentScreen("intro");
    setRoundResults([]);
    setMockData(null);
    setOutcome(null);
  };

  const handleNext = () => {
    next();
  };

  // Build allPlayers array for components
  const buildAllPlayers = () => {
    if (!mockData) return [];
    return mockData.players.map((p, idx) => ({
      type: idx === 2 ? "real" : "virtual",
      player: p,
      playerId: p.playerId,
      bot: idx === 2 ? null : { stats: p.stats, playerId: p.playerId }
    }));
  };

  // Intro screen
  if (currentScreen === "intro") {
    return (
      <div className="p-8 max-w-4xl mx-auto">
        <h3 className="text-2xl font-bold text-gray-900 mb-4 text-center">
          Tutorial Round 1
        </h3>

        <div className="bg-yellow-50 border border-yellow-300 rounded-lg p-4 mb-6">
          <p className="text-sm text-yellow-900 mb-2">
            In this round, you can see what roles your teammates have already chosen before you make your decision.
          </p>
          <p className="text-sm text-yellow-900 font-semibold">
            Notice that the other two players have chosen the Fighter roles.
            What role do you think you should play, given their choices?
          </p>
          <p className="text-sm text-yellow-900 mt-2">
            Remember: You'll be committed to your chosen role for 2 rounds, just like in the main game!
          </p>
        </div>

        <div className="flex justify-center">
          <Button handleClick={handleStartTutorial} autoFocus>
            <p>Begin Tutorial Round 1</p>
          </Button>
        </div>
      </div>
    );
  }

  // Role selection screen
  if (currentScreen === "selection") {
    return (
      <div className="p-8 max-w-4xl mx-auto">
        <h3 className="text-2xl font-bold text-gray-900 mb-4 text-center">
          Tutorial Round 1 - Role Selection
        </h3>

        <div className="bg-yellow-50 border border-yellow-300 rounded-lg p-4 mb-6">
          <p className="text-sm text-yellow-900 font-semibold">
            Your teammates' roles are shown below. Choose a role that complements the team!
          </p>
        </div>

        <div className="bg-white border-2 border-gray-200 rounded-lg p-6 mb-6">
          <h4 className="font-semibold mb-4 text-center">Your Teammates' Roles:</h4>
          <div className="flex gap-6 justify-center mb-6">
            <div className="text-center bg-red-50 border-2 border-red-300 rounded p-4">
              <div className="text-4xl mb-2">⚔️</div>
              <div className="text-sm font-semibold">Player 1</div>
              <div className="text-lg font-bold text-red-700">Fighter</div>
            </div>
            <div className="text-center bg-red-50 border-2 border-red-300 rounded p-4">
              <div className="text-4xl mb-2">⚔️</div>
              <div className="text-sm font-semibold">Player 2</div>
              <div className="text-lg font-bold text-red-700">Fighter</div>
            </div>
          </div>

          <div className="border-t-2 border-gray-300 pt-6">
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
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="flex justify-center">
          <Button
            handleClick={handleConfirm}
            disabled={selectedRole === null}
          >
            <p>Confirm Role & Start Battle</p>
          </Button>
        </div>
      </div>
    );
  }

  // Round playback screens (round1, round2)
  if (currentScreen === "round1" || currentScreen === "round2") {
    const roundIndex = currentScreen === "round1" ? 0 : 1;
    const result = roundResults[roundIndex];
    const allPlayers = buildAllPlayers();

    return (
      <MockDataProvider mockData={mockData}>
        <div className="fixed inset-0 bg-gradient-to-b from-blue-400 to-blue-600 flex items-center justify-center p-2">
          <div className="w-full h-full flex items-center justify-center" style={{ maxWidth: '1400px' }}>
            <div className="bg-white rounded-lg shadow-2xl border-4 border-gray-800 w-full h-full flex flex-col overflow-hidden">
              {/* Header */}
              <div className="bg-gray-800 text-white text-center py-3">
                <h1 className="text-xl font-bold">Tutorial - Round {result.roundNum} of 2</h1>
              </div>

              {/* Content */}
              <div className="flex-1 flex flex-col p-6 overflow-auto">
                {/* Battle Field */}
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

                {/* Results Panel */}
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

              {/* Footer Button */}
              <div className="bg-gray-100 border-t-4 border-gray-700 p-4 flex justify-center">
                <Button handleClick={handleNextRound} autoFocus>
                  <p>{currentScreen === "round1" && roundResults.length > 1 ? `Continue to Round 2` : "See Final Results"}</p>
                </Button>
              </div>
            </div>
          </div>
        </div>
      </MockDataProvider>
    );
  }

  // Final summary screen
  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h3 className="text-2xl font-bold text-gray-900 mb-4 text-center">
        Tutorial Round 1 - Final Results
      </h3>

      <div className={`border-2 rounded-lg p-6 mb-6 ${outcome.success ? 'bg-green-50 border-green-300' : 'bg-orange-50 border-orange-300'}`}>
        <p className="text-lg font-semibold mb-2 text-center">
          {outcome.message}
        </p>
      </div>

      <div className="bg-gray-50 rounded-lg p-6 mb-6">
        <h4 className="font-semibold mb-3">Final State:</h4>
        <div className="grid grid-cols-2 gap-4">
          <div className="text-center">
            <div className="text-sm text-gray-600">Enemy Health</div>
            <div className={`text-3xl font-bold ${outcome.enemyHealth === 0 ? 'text-gray-400 line-through' : 'text-red-600'}`}>
              {outcome.enemyHealth}
            </div>
            {outcome.enemyHealth === 0 && <div className="text-xs text-green-600 font-semibold mt-1">DEFEATED!</div>}
          </div>
          <div className="text-center">
            <div className="text-sm text-gray-600">Team Health</div>
            <div className={`text-3xl font-bold ${outcome.teamHealth === 0 ? 'text-gray-400 line-through' : 'text-green-600'}`}>
              {outcome.teamHealth}
            </div>
            {outcome.teamHealth === 0 && <div className="text-xs text-red-600 font-semibold mt-1">DEFEATED!</div>}
          </div>
        </div>
      </div>

      <div className="bg-blue-50 border border-blue-200 rounded p-4 mb-6">
        <p className="text-sm text-blue-800">
          <strong>Key Learning:</strong> When you know what roles your teammates have chosen,
          you can pick a complementary role to maximize your team's effectiveness.
          {outcome.success
            ? " With two Fighters already attacking, adding a Tank or Healer created a balanced team that won in just 2 rounds!"
            : " With three Fighters, the team had high damage but no defense or healing, leading to defeat."}
        </p>
        <p className="text-sm text-blue-800 mt-2">
          Notice how your role choice stayed locked for all {roundResults.length} rounds - this is how the main game works too!
        </p>
      </div>

      <div className="flex justify-center gap-4">
        <Button handleClick={handlePlayAgain}>
          <p>Play Again</p>
        </Button>
        <Button handleClick={handleNext} autoFocus>
          <p>Continue to Tutorial Round 2</p>
        </Button>
      </div>
    </div>
  );
}

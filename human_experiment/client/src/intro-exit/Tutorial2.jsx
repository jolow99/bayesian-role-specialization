import React, { useState } from "react";
import { Button } from "../components/Button";

const ROLES = { FIGHTER: 0, TANK: 1, HEALER: 2 };
const ROLE_NAMES = ["Fighter", "Tank", "Healer"];
const ROLE_ICONS = ["⚔️", "🛡️", "💚"];
const ACTION_NAMES = ["Attack", "Defend", "Heal"];
const ACTION_ICONS = ["⚔️", "🛡️", "💚"];

export function Tutorial2({ next }) {
  const [selectedRole, setSelectedRole] = useState(null);
  const [currentScreen, setCurrentScreen] = useState("intro"); // intro, round1, selection, round2, round3, final
  const [roundResults, setRoundResults] = useState([]);
  const [enemyHealth, setEnemyHealth] = useState(10);
  const [teamHealth, setTeamHealth] = useState(10);
  const [outcome, setOutcome] = useState(null);

  // Bot players: One Fighter (always attacks), One Tank (defends when enemy attacks)
  const actualBotRoles = [ROLES.FIGHTER, ROLES.TANK];

  const getBotAction = (role, enemyAttacks) => {
    if (role === ROLES.FIGHTER) return "Attack";
    if (role === ROLES.TANK) return enemyAttacks ? "Defend" : "Attack";
    if (role === ROLES.HEALER) return "Heal"; // Won't be used in this tutorial
    return "Attack";
  };

  // Simulate a single round of combat
  const simulateRound = (roundNum, playerRole, currentEnemyHP, currentTeamHP, fixedEnemyAttacks = null) => {
    // Determine if enemy attacks (fixed for round 1, random otherwise)
    const enemyAttacks = fixedEnemyAttacks !== null ? fixedEnemyAttacks : Math.random() > 0.5;

    // Determine player stats (simplified - all balanced for tutorial)
    const STR = 0.33, DEF = 0.33, SUP = 0.33;

    // Get bot actions
    const bot1Action = getBotAction(actualBotRoles[0], enemyAttacks);
    const bot2Action = getBotAction(actualBotRoles[1], enemyAttacks);

    // Determine player action based on role
    let playerAction;
    if (playerRole === ROLES.FIGHTER) playerAction = "Attack";
    else if (playerRole === ROLES.TANK) playerAction = enemyAttacks ? "Defend" : "Attack";
    else if (playerRole === ROLES.HEALER) playerAction = (currentTeamHP <= 5) ? "Heal" : "Attack";
    else playerAction = "Attack";

    // Calculate damage to enemy
    let attackCount = [bot1Action, bot2Action, playerAction].filter(a => a === "Attack").length;
    let damageToEnemy = attackCount * STR * 1.5;

    // Calculate damage to team
    let hasDefender = [bot1Action, bot2Action, playerAction].includes("Defend");
    let damageToTeam = 0;
    if (enemyAttacks) {
      const bossDamage = 2; // Simplified boss damage
      if (hasDefender) {
        damageToTeam = Math.max(0, bossDamage - (DEF * 3));
      } else {
        damageToTeam = bossDamage;
      }
    }

    // Calculate healing
    let healCount = [bot1Action, bot2Action, playerAction].filter(a => a === "Heal").length;
    let healAmount = healCount * SUP * 2;

    // Update health
    const newEnemyHP = Math.max(0, currentEnemyHP - damageToEnemy);
    const newTeamHP = Math.max(0, Math.min(10, currentTeamHP - damageToTeam + healAmount));

    return {
      roundNum,
      enemyAttacks,
      bot1Action,
      bot2Action,
      playerAction,
      playerRole,
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
    const round1Result = simulateRound(1, null, 10, 10, true); // true = enemy attacks
    setRoundResults([round1Result]);
    setEnemyHealth(round1Result.enemyHealth);
    setTeamHealth(round1Result.teamHealth);
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

      // Check if game ended early
      if (currentEnemyHP <= 0 || currentTeamHP <= 0) {
        break;
      }
    }

    setRoundResults(results);
    setEnemyHealth(currentEnemyHP);
    setTeamHealth(currentTeamHP);

    // Determine outcome
    let outcomeMessage;
    let success = false;
    if (currentEnemyHP <= 0) {
      outcomeMessage = "Victory! Great job coordinating with your team!";
      success = true;
    } else if (currentTeamHP <= 0) {
      outcomeMessage = "Defeat! Your team was overwhelmed.";
    } else {
      // Neither died - evaluate based on final state and role choice
      if (selectedRole === ROLES.HEALER) {
        outcomeMessage = "Excellent! Your healing kept the team healthy while they attacked and defended.";
        success = true;
      } else if (selectedRole === ROLES.TANK) {
        outcomeMessage = "The team survived, but having two defenders is redundant. A healer would have been more helpful!";
      } else {
        outcomeMessage = "The team survived with heavy damage. Without healing, it was harder to stay healthy!";
      }
    }

    setOutcome({ message: outcomeMessage, success });
    setCurrentScreen("round2");
  };

  const handleNextRound = () => {
    const currentRoundNum = parseInt(currentScreen.replace("round", ""));
    if (currentRoundNum < roundResults.length) {
      setCurrentScreen(`round${currentRoundNum + 1}`);
    } else {
      setCurrentScreen("final");
    }
  };

  const handleNext = () => {
    next();
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

    return (
      <div className="p-8 max-w-4xl mx-auto">
        <h3 className="text-2xl font-bold text-gray-900 mb-4 text-center">
          Tutorial Round 2 - Observing Round 1
        </h3>

        <div className="bg-yellow-50 border border-yellow-300 rounded-lg p-4 mb-6">
          <p className="text-sm text-yellow-900 mb-2">
            Here's what happened in the first round. Pay attention to what actions your teammates took!
          </p>
          <p className="text-sm text-yellow-900 font-semibold">
            Based on their actions, what roles do you think they're playing? What role would complement the team best?
          </p>
        </div>

        <div className="bg-gray-50 rounded-lg p-6 mb-6">
          <h4 className="font-semibold mb-3 text-center">Round 1 Results:</h4>

          {/* Team Actions */}
          <div className="mb-4 pb-4 border-b border-gray-300">
            <p className="text-sm text-gray-700 mb-2 text-center"><strong>What Each Player Did:</strong></p>
            <div className="flex gap-4 justify-center">
              <div className="text-center bg-gray-100 rounded p-3">
                <div className="text-3xl mb-1">{ACTION_ICONS[0]}</div>
                <div className="text-xs text-gray-600">Player 1</div>
                <div className="text-sm font-semibold">{result.bot1Action}</div>
              </div>
              <div className="text-center bg-gray-100 rounded p-3">
                <div className="text-3xl mb-1">{ACTION_ICONS[result.bot2Action === "Attack" ? 0 : 1]}</div>
                <div className="text-xs text-gray-600">Player 2</div>
                <div className="text-sm font-semibold">{result.bot2Action}</div>
              </div>
              <div className="text-center bg-gray-200 rounded p-3 opacity-50">
                <div className="text-3xl mb-1">❓</div>
                <div className="text-xs text-gray-600">You</div>
                <div className="text-sm font-semibold">Not yet chosen</div>
              </div>
            </div>
          </div>

          {/* Health Changes */}
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div className="bg-red-50 border-2 border-red-400 rounded-lg p-4">
              <div className="text-xs font-semibold text-gray-600 mb-2 uppercase tracking-wide">Enemy Health</div>
              <div className="flex items-center justify-center gap-2 mb-2">
                <span className="text-2xl font-bold text-gray-700">{result.previousEnemyHealth}</span>
                <span className="text-xl text-gray-400">→</span>
                <span className="text-2xl font-bold text-red-600">{result.enemyHealth}</span>
              </div>
              {result.damageToEnemy > 0 && (
                <div className="text-sm font-medium text-red-600">
                  -{result.damageToEnemy} damage dealt
                </div>
              )}
            </div>

            <div className="bg-green-50 border-2 border-green-400 rounded-lg p-4">
              <div className="text-xs font-semibold text-gray-600 mb-2 uppercase tracking-wide">Team Health</div>
              <div className="flex items-center justify-center gap-2 mb-2">
                <span className="text-2xl font-bold text-gray-700">{result.previousTeamHealth}</span>
                <span className="text-xl text-gray-400">→</span>
                <span className={`text-2xl font-bold ${result.teamHealth > result.previousTeamHealth ? 'text-green-600' : result.teamHealth < result.previousTeamHealth ? 'text-orange-600' : 'text-gray-700'}`}>
                  {result.teamHealth}
                </span>
              </div>
              <div className="text-sm font-medium space-y-1">
                {result.damageToTeam > 0 && (
                  <div className="text-orange-600">-{result.damageToTeam} damage taken</div>
                )}
                {result.damageToTeam === 0 && <div className="text-gray-500">No damage (defended!)</div>}
              </div>
            </div>
          </div>

          {/* Enemy Action */}
          <div className="bg-blue-50 border border-blue-200 rounded p-3 text-center">
            <div className="text-sm text-gray-600 mb-1">Enemy Action:</div>
            <div className="flex items-center justify-center gap-2">
              <div className="text-3xl">{result.enemyAttacks ? "⚔️" : "😴"}</div>
              <div className="text-sm font-medium text-gray-700">
                {result.enemyAttacks ? "Enemy attacked!" : "Enemy rested"}
              </div>
            </div>
          </div>
        </div>

        <div className="flex justify-center">
          <Button handleClick={() => setCurrentScreen("selection")} autoFocus>
            <p>Choose Your Role</p>
          </Button>
        </div>
      </div>
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
                <div className="text-4xl mb-2">{ROLE_ICONS[role]}</div>
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

    return (
      <div className="p-8 max-w-4xl mx-auto">
        <h3 className="text-2xl font-bold text-gray-900 mb-4 text-center">
          Tutorial - Round {result.roundNum} of 3
        </h3>

        <div className="bg-gray-50 rounded-lg p-6 mb-6">
          <h4 className="font-semibold mb-3 text-center">What Happened This Round:</h4>

          {/* Team Actions */}
          <div className="mb-4 pb-4 border-b border-gray-300">
            <p className="text-sm text-gray-700 mb-2 text-center"><strong>What Each Player Did:</strong></p>
            <div className="flex gap-4 justify-center">
              <div className="text-center bg-gray-100 rounded p-3">
                <div className="text-3xl mb-1">⚔️</div>
                <div className="text-xs text-gray-600">P1</div>
                <div className="text-sm font-semibold">{result.bot1Action}</div>
              </div>
              <div className="text-center bg-gray-100 rounded p-3">
                <div className="text-3xl mb-1">{result.bot2Action === "Attack" ? "⚔️" : "🛡️"}</div>
                <div className="text-xs text-gray-600">P2</div>
                <div className="text-sm font-semibold">{result.bot2Action}</div>
              </div>
              <div className="text-center bg-blue-100 border-2 border-blue-400 rounded p-3">
                <div className="text-3xl mb-1">{ROLE_ICONS[selectedRole]}</div>
                <div className="text-xs text-gray-600">You</div>
                <div className="text-sm font-semibold">{result.playerAction}</div>
              </div>
            </div>
          </div>

          {/* Health Changes */}
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div className="bg-red-50 border-2 border-red-400 rounded-lg p-4">
              <div className="text-xs font-semibold text-gray-600 mb-2 uppercase tracking-wide">Enemy Health</div>
              <div className="flex items-center justify-center gap-2 mb-2">
                <span className="text-2xl font-bold text-gray-700">{result.previousEnemyHealth}</span>
                <span className="text-xl text-gray-400">→</span>
                <span className="text-2xl font-bold text-red-600">{result.enemyHealth}</span>
              </div>
              {result.damageToEnemy > 0 && (
                <div className="text-sm font-medium text-red-600">
                  -{result.damageToEnemy} damage dealt
                </div>
              )}
            </div>

            <div className="bg-green-50 border-2 border-green-400 rounded-lg p-4">
              <div className="text-xs font-semibold text-gray-600 mb-2 uppercase tracking-wide">Team Health</div>
              <div className="flex items-center justify-center gap-2 mb-2">
                <span className="text-2xl font-bold text-gray-700">{result.previousTeamHealth}</span>
                <span className="text-xl text-gray-400">→</span>
                <span className={`text-2xl font-bold ${result.teamHealth > result.previousTeamHealth ? 'text-green-600' : result.teamHealth < result.previousTeamHealth ? 'text-orange-600' : 'text-gray-700'}`}>
                  {result.teamHealth}
                </span>
              </div>
              <div className="text-sm font-medium space-y-1">
                {result.damageToTeam > 0 && (
                  <div className="text-orange-600">-{result.damageToTeam} damage taken</div>
                )}
                {result.healAmount > 0 && (
                  <div className="text-green-600">+{result.healAmount} healing received</div>
                )}
                {result.damageToTeam === 0 && result.healAmount === 0 && (
                  <div className="text-gray-500">No change</div>
                )}
              </div>
            </div>
          </div>

          {/* Enemy Action */}
          <div className="bg-blue-50 border border-blue-200 rounded p-3 text-center">
            <div className="text-sm text-gray-600 mb-1">Enemy Action:</div>
            <div className="flex items-center justify-center gap-2">
              <div className="text-3xl">{result.enemyAttacks ? "⚔️" : "😴"}</div>
              <div className="text-sm font-medium text-gray-700">
                {result.enemyAttacks ? "Enemy attacked!" : "Enemy rested"}
              </div>
            </div>
          </div>
        </div>

        <div className="flex justify-center">
          <Button handleClick={handleNextRound} autoFocus>
            <p>{roundNum < 3 ? `Continue to Round ${roundNum + 1}` : "See Final Results"}</p>
          </Button>
        </div>
      </div>
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
            <div className="text-3xl font-bold text-red-600">{enemyHealth}</div>
          </div>
          <div className="text-center">
            <div className="text-sm text-gray-600">Team Health</div>
            <div className="text-3xl font-bold text-green-600">{teamHealth}</div>
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
              <div className="text-3xl mb-1">{ROLE_ICONS[selectedRole]}</div>
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

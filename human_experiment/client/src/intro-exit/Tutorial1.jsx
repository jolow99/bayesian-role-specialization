import React, { useState } from "react";
import { Button } from "../components/Button";

const ROLES = { FIGHTER: 0, TANK: 1, HEALER: 2 };
const ROLE_NAMES = ["Fighter", "Tank", "Healer"];
const ROLE_ICONS = ["⚔️", "🛡️", "💚"];

export function Tutorial1({ next }) {
  const [selectedRole, setSelectedRole] = useState(null);
  const [currentRound, setCurrentRound] = useState(0); // 0 = selection, 1-3 = rounds, 4 = final
  const [roundResults, setRoundResults] = useState([]);
  const [enemyHealth, setEnemyHealth] = useState(10);
  const [teamHealth, setTeamHealth] = useState(10);
  const [outcome, setOutcome] = useState(null);

  // Bot players both chose Fighter (indices 0 and 1)
  const botRoles = [ROLES.FIGHTER, ROLES.FIGHTER];

  const handleRoleSelect = (role) => {
    setSelectedRole(role);
  };

  // Simulate a single round of combat
  const simulateRound = (roundNum, playerRole, currentEnemyHP, currentTeamHP) => {
    // Enemy always attacks in tutorial for predictable outcomes
    const enemyAttacks = true;

    // Calculate team actions based on roles
    const roles = [...botRoles, playerRole];

    // Count attacks - 2 Fighters deal 5 damage each, 3 Fighters deal 3.5 damage each
    let attackCount = roles.filter(r => r === ROLES.FIGHTER).length;
    let damageToEnemy;
    if (attackCount === 2) {
      damageToEnemy = 5; // 2 fighters = 5 damage per round (kills boss in 2 rounds)
    } else if (attackCount === 3) {
      damageToEnemy = 3.5; // 3 fighters = 3.5 damage (not enough without defense/healing)
    } else {
      damageToEnemy = 0; // No fighters (shouldn't happen in this tutorial)
    }

    // Check for defense
    let hasDefender = roles.includes(ROLES.TANK);
    let damageToTeam = 0;
    const bossDamage = 6; // Boss deals 6 damage per attack
    if (hasDefender) {
      damageToTeam = 1; // Tank reduces it to 1 damage
    } else {
      damageToTeam = bossDamage; // Full damage without tank
    }

    // Check for healing
    let hasHealer = roles.includes(ROLES.HEALER);
    let healAmount = 0;
    if (hasHealer && currentTeamHP < 10) {
      healAmount = 3; // Healer restores 3 HP
    }

    // Update health
    const newEnemyHP = Math.max(0, currentEnemyHP - damageToEnemy);
    const newTeamHP = Math.max(0, Math.min(10, currentTeamHP - damageToTeam + healAmount));

    return {
      roundNum,
      enemyAttacks,
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

  const handleConfirm = () => {
    // Simulate rounds until game ends (max 2 rounds)
    const results = [];
    let currentEnemyHP = 10;
    let currentTeamHP = 10;

    for (let i = 1; i <= 2; i++) {
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
      outcomeMessage = "Victory! By choosing a Tank or Healer role, you complemented the two Fighters perfectly!";
      success = true;
    } else if (currentTeamHP <= 0) {
      outcomeMessage = "Defeat! With three Fighters and no Tank or Healer, your team couldn't survive the enemy's attacks. Try a different role!";
    } else {
      // Shouldn't happen with new mechanics, but just in case
      outcomeMessage = "The battle continues... Consider choosing Tank or Healer to complement your Fighter teammates!";
    }

    setOutcome({ message: outcomeMessage, success });
    setCurrentRound(1);
  };

  const handleNextRound = () => {
    if (currentRound < roundResults.length) {
      setCurrentRound(currentRound + 1);
    } else {
      setCurrentRound(4); // Move to final summary
    }
  };

  const handleNext = () => {
    next();
  };

  const handlePlayAgain = () => {
    // Reset all state to initial values
    setSelectedRole(null);
    setCurrentRound(0);
    setRoundResults([]);
    setEnemyHealth(10);
    setTeamHealth(10);
    setOutcome(null);
  };

  // Selection screen
  if (currentRound === 0) {
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

        <div className="bg-white border-2 border-gray-200 rounded-lg p-6 mb-6">
          <h4 className="font-semibold mb-4 text-center">Your Teammates' Roles:</h4>
          <div className="flex gap-6 justify-center mb-6">
            <div className="text-center bg-red-50 border-2 border-red-300 rounded p-4">
              <div className="text-4xl mb-2">{ROLE_ICONS[ROLES.FIGHTER]}</div>
              <div className="text-sm font-semibold">Player 1</div>
              <div className="text-lg font-bold text-red-700">Fighter</div>
            </div>
            <div className="text-center bg-red-50 border-2 border-red-300 rounded p-4">
              <div className="text-4xl mb-2">{ROLE_ICONS[ROLES.FIGHTER]}</div>
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
                  <div className="text-4xl mb-2">{ROLE_ICONS[role]}</div>
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

  // Round-by-round playback (rounds 1-3)
  if (currentRound > 0 && currentRound <= roundResults.length) {
    const result = roundResults[currentRound - 1];

    return (
      <div className="p-8 max-w-4xl mx-auto">
        <h3 className="text-2xl font-bold text-gray-900 mb-4 text-center">
          Tutorial - Round {result.roundNum} of {roundResults.length}
        </h3>

        <div className="bg-gray-50 rounded-lg p-6 mb-6">
          <h4 className="font-semibold mb-3 text-center">What Happened This Round:</h4>

          {/* Team Composition */}
          <div className="mb-4 pb-4 border-b border-gray-300">
            <p className="text-sm text-gray-700 mb-2 text-center"><strong>Team Roles (Locked for the battle):</strong></p>
            <div className="flex gap-4 justify-center">
              <div className="text-center bg-gray-100 rounded p-3">
                <div className="text-3xl mb-1">{ROLE_ICONS[ROLES.FIGHTER]}</div>
                <div className="text-xs text-gray-600">P1: Fighter</div>
              </div>
              <div className="text-center bg-gray-100 rounded p-3">
                <div className="text-3xl mb-1">{ROLE_ICONS[ROLES.FIGHTER]}</div>
                <div className="text-xs text-gray-600">P2: Fighter</div>
              </div>
              <div className="text-center bg-blue-100 border-2 border-blue-400 rounded p-3">
                <div className="text-3xl mb-1">{ROLE_ICONS[selectedRole]}</div>
                <div className="text-xs text-gray-600">You: {ROLE_NAMES[selectedRole]}</div>
              </div>
            </div>
          </div>

          {/* Actions and Results */}
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
            <p>{currentRound < roundResults.length ? `Continue to Round ${currentRound + 1}` : "See Final Results"}</p>
          </Button>
        </div>
      </div>
    );
  }

  // Final summary (round 4)
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
            <div className={`text-3xl font-bold ${enemyHealth === 0 ? 'text-gray-400 line-through' : 'text-red-600'}`}>{enemyHealth}</div>
            {enemyHealth === 0 && <div className="text-xs text-green-600 font-semibold mt-1">DEFEATED!</div>}
          </div>
          <div className="text-center">
            <div className="text-sm text-gray-600">Team Health</div>
            <div className={`text-3xl font-bold ${teamHealth === 0 ? 'text-gray-400 line-through' : 'text-green-600'}`}>{teamHealth}</div>
            {teamHealth === 0 && <div className="text-xs text-red-600 font-semibold mt-1">DEFEATED!</div>}
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

import React from "react";
import { useGame, useRound, usePlayers } from "@empirica/core/player/classic/react";
import { HealthBar } from "../components/HealthBar";

export function Reveal() {
  const game = useGame();
  const round = useRound();
  const players = usePlayers();

  const actions = round.get("actions") || [];
  const enemyHealth = game.get("enemyHealth") || 10;
  const teamHealth = game.get("teamHealth") || 10;
  const previousEnemyHealth = round.get("enemyHealth") || 10;
  const previousTeamHealth = round.get("teamHealth") || 10;
  const damageToEnemy = round.get("damageToEnemy") || 0;
  const damageToTeam = round.get("damageToTeam") || 0;
  const healAmount = round.get("healAmount") || 0;
  const roundNumber = round.get("roundNumber");
  const maxRounds = game.get("maxRounds");
  const maxHealth = game.get("maxHealth") || 10;

  const actionIcons = {
    ATTACK: "⚔️",
    DEFEND: "🛡️",
    HEAL: "💚"
  };

  const actionLabels = {
    ATTACK: "Attack",
    DEFEND: "Defend",
    HEAL: "Heal"
  };

  return (
    <div className="w-full max-w-5xl mx-auto p-6">
      <div className="bg-white border-2 border-gray-300 rounded-lg p-6 shadow-lg">
        <div className="text-center mb-6">
          <h2 className="text-3xl font-bold text-gray-800 mb-2">
            Round {roundNumber} Results
          </h2>
        </div>

        {/* Player Actions */}
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-gray-700 mb-3 text-center">
            Player Actions
          </h3>
          <div className="flex justify-center gap-4">
            {Array.isArray(actions) ? (
              actions.map((action, idx) => (
                <div
                  key={idx}
                  className="bg-gray-50 border-2 border-gray-300 rounded-lg p-4 text-center min-w-[120px]"
                >
                  <div className="text-3xl mb-2">{actionIcons[action]}</div>
                  <div className="text-sm font-semibold text-gray-700">Player {idx + 1}</div>
                  <div className="text-xs text-gray-600">{actionLabels[action]}</div>
                </div>
              ))
            ) : (
              <div className="text-gray-500 text-sm">Waiting for actions...</div>
            )}
          </div>
        </div>

        {/* Combat Results */}
        <div className="grid grid-cols-2 gap-6 mb-6">
          {/* Enemy Damage */}
          <div className="bg-red-50 border-2 border-red-300 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-red-700 mb-3 text-center">
              Enemy Status
            </h3>
            <HealthBar
              label="Enemy Health"
              current={enemyHealth}
              max={maxHealth}
              color="red"
            />
            {damageToEnemy > 0 && (
              <div className="mt-3 text-center">
                <span className="text-red-600 font-bold text-lg">
                  -{damageToEnemy} damage dealt!
                </span>
              </div>
            )}
          </div>

          {/* Team Status */}
          <div className="bg-green-50 border-2 border-green-300 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-green-700 mb-3 text-center">
              Team Status
            </h3>
            <HealthBar
              label="Team Health"
              current={teamHealth}
              max={maxHealth}
              color="green"
            />
            <div className="mt-3 space-y-1">
              {damageToTeam > 0 && (
                <div className="text-center text-red-600 font-bold">
                  -{damageToTeam} damage taken
                </div>
              )}
              {healAmount > 0 && (
                <div className="text-center text-green-600 font-bold">
                  +{healAmount} health restored
                </div>
              )}
              {damageToTeam === 0 && healAmount === 0 && (
                <div className="text-center text-gray-500 text-sm">
                  No change
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Progress Indicator */}
        <div className="text-center">
          <div className="text-sm text-gray-600 mb-2">
            Round {roundNumber} of {maxRounds}
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
            <div
              className="bg-empirica-500 h-full transition-all duration-500"
              style={{ width: `${(roundNumber / maxRounds) * 100}%` }}
            />
          </div>
        </div>

        {/* Status Message */}
        {enemyHealth <= 0 && (
          <div className="mt-4 p-4 bg-green-100 border-2 border-green-500 rounded-lg text-center">
            <div className="text-2xl mb-2">🎉</div>
            <div className="text-xl font-bold text-green-700">Victory! Enemy Defeated!</div>
          </div>
        )}

        {teamHealth <= 0 && (
          <div className="mt-4 p-4 bg-red-100 border-2 border-red-500 rounded-lg text-center">
            <div className="text-2xl mb-2">💀</div>
            <div className="text-xl font-bold text-red-700">Defeat! Team Eliminated!</div>
          </div>
        )}

        {enemyHealth > 0 && teamHealth > 0 && (
          <div className="mt-4 text-center text-gray-600 text-sm">
            Preparing next round...
          </div>
        )}
      </div>
    </div>
  );
}

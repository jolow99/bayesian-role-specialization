import React from "react";
import { ACTION_ICONS } from "../constants";

const actionIcons = ACTION_ICONS;

export const ResultsPanel = React.memo(function ResultsPanel({
  stageNumber,
  turnNumber,
  actions = [],
  allPlayers = [],
  currentPlayerGameId,
  enemyIntent,
  onNextTurn,
  nextButtonLabel = "Next Turn",
  // Health transition data
  previousTeamHealth,
  newTeamHealth,
  previousEnemyHealth,
  newEnemyHealth,
  damageToTeam = 0,
  damageToEnemy = 0,
  healAmount = 0
}) {
  // Calculate health changes
  const teamHealthChange = (newTeamHealth ?? 0) - (previousTeamHealth ?? 0);
  const enemyHealthChange = (newEnemyHealth ?? 0) - (previousEnemyHealth ?? 0);
  return (
    <div className="animate-fadeIn">
      <div className="bg-gray-800 text-white rounded-lg px-4 py-3 text-center mb-4">
        <h3 className="text-xl font-bold">
          Stage {stageNumber} - Turn {turnNumber} Results
        </h3>
      </div>

      {/* Two-column Action Summary */}
      <div className="flex gap-4 mb-4">
        {/* Left: Team Side */}
        <div className="flex-1 bg-green-50 border-2 border-green-300 rounded-lg p-4">
          <div className="text-xs font-semibold text-green-700 mb-2 uppercase tracking-wide text-center">Your Team</div>

          {/* Team HP Transition */}
          <div className="bg-white rounded-lg p-2 mb-3 border border-green-200">
            <div className="flex items-center justify-center gap-2 text-lg font-bold">
              <span className="text-green-600">👥 {previousTeamHealth ?? '?'}</span>
              <span className="text-gray-400">→</span>
              <span className="text-green-700">{newTeamHealth ?? '?'}</span>
              {teamHealthChange !== 0 && (
                <span className={`text-sm ${teamHealthChange > 0 ? 'text-green-500' : 'text-red-500'}`}>
                  ({teamHealthChange > 0 ? '+' : ''}{teamHealthChange})
                </span>
              )}
            </div>
            {/* Breakdown of changes */}
            <div className="flex justify-center gap-3 mt-1 text-xs">
              {damageToTeam > 0 && (
                <span className="text-red-500">-{damageToTeam} dmg</span>
              )}
              {healAmount > 0 && (
                <span className="text-green-500">+{healAmount} heal</span>
              )}
            </div>
          </div>

          {/* Team Actions */}
          <div className="flex justify-center gap-3">
            {actions.map((action, playerId) => {
              const entry = allPlayers[playerId];
              if (!entry) return null;
              const isCurrentPlayer = playerId === currentPlayerGameId;

              return (
                <div key={playerId} className="flex flex-col items-center bg-white rounded-lg px-3 py-2 border border-green-200">
                  <div className="text-3xl mb-1">{actionIcons[action]}</div>
                  <div className={`text-xs font-semibold ${isCurrentPlayer ? 'text-blue-600' : 'text-gray-600'}`}>
                    {isCurrentPlayer ? "YOU" : `P${playerId + 1}`}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right: Enemy Side */}
        <div className="flex-1 bg-red-50 border-2 border-red-300 rounded-lg p-4">
          <div className="text-xs font-semibold text-red-700 mb-2 uppercase tracking-wide text-center">Enemy</div>

          {/* Enemy HP Transition */}
          <div className="bg-white rounded-lg p-2 mb-3 border border-red-200">
            <div className="flex items-center justify-center gap-2 text-lg font-bold">
              <span className="text-red-600">👹 {previousEnemyHealth ?? '?'}</span>
              <span className="text-gray-400">→</span>
              <span className="text-red-700">{newEnemyHealth ?? '?'}</span>
              {enemyHealthChange !== 0 && (
                <span className="text-red-500 text-sm">
                  ({enemyHealthChange})
                </span>
              )}
            </div>
            {/* Breakdown of changes */}
            {damageToEnemy > 0 && (
              <div className="text-center mt-1 text-xs text-red-500">
                -{damageToEnemy} dmg
              </div>
            )}
          </div>

          {/* Enemy Action */}
          <div className="flex flex-col items-center bg-white rounded-lg px-4 py-2 border border-red-200">
            <div className="text-4xl mb-1">{enemyIntent === "WILL_ATTACK" ? "⚔️" : "😴"}</div>
            <div className="text-xs font-semibold text-gray-600">
              {enemyIntent === "WILL_ATTACK" ? "Attacked!" : "Rested"}
            </div>
          </div>
        </div>
      </div>

      <div className="text-center">
        <button
          onClick={onNextTurn}
          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 text-white font-bold rounded-lg shadow-lg transition-colors"
        >
          {nextButtonLabel} →
        </button>
      </div>
    </div>
  );
});

import React from "react";

const actionIcons = {
  ATTACK: "⚔️",
  DEFEND: "🛡️",
  HEAL: "💚"
};

export const ResultsPanel = React.memo(function ResultsPanel({
  roundNumber,
  enemyHealth,
  previousEnemyHealth,
  damageToEnemy,
  teamHealth,
  previousTeamHealth,
  damageToTeam,
  healAmount,
  actions = [],
  allPlayers = [],
  currentPlayerId,
  enemyIntent,
  countdown
}) {
  return (
    <div className="animate-fadeIn">
      <div className="bg-gray-800 text-white rounded-lg px-4 py-3 text-center mb-4">
        <h3 className="text-xl font-bold">Round {roundNumber} Results</h3>
      </div>

      {/* Health Changes Summary */}
      <div className="grid grid-cols-2 gap-4 mb-4">
        {/* Enemy Health Change */}
        <div className="bg-red-50 border-2 border-red-400 rounded-lg p-4">
          <div className="text-xs font-semibold text-gray-600 mb-2 uppercase tracking-wide">Enemy Health</div>
          <div className="flex items-center justify-center gap-2 mb-2">
            <span className="text-2xl font-bold text-gray-700">{previousEnemyHealth}</span>
            <span className="text-xl text-gray-400">→</span>
            <span className="text-2xl font-bold text-red-600">{enemyHealth}</span>
          </div>
          {damageToEnemy > 0 && (
            <div className="text-sm font-medium text-red-600">
              -{damageToEnemy} damage dealt
            </div>
          )}
        </div>

        {/* Team Health Change */}
        <div className="bg-green-50 border-2 border-green-400 rounded-lg p-4">
          <div className="text-xs font-semibold text-gray-600 mb-2 uppercase tracking-wide">Team Health</div>
          <div className="flex items-center justify-center gap-2 mb-2">
            <span className="text-2xl font-bold text-gray-700">{previousTeamHealth}</span>
            <span className="text-xl text-gray-400">→</span>
            <span className={`text-2xl font-bold ${teamHealth > previousTeamHealth ? 'text-green-600' : teamHealth < previousTeamHealth ? 'text-orange-600' : 'text-gray-700'}`}>
              {teamHealth}
            </span>
          </div>
          <div className="text-sm font-medium space-y-1">
            {damageToTeam > 0 && (
              <div className="text-orange-600">-{damageToTeam} damage taken</div>
            )}
            {healAmount > 0 && (
              <div className="text-green-600">+{healAmount} healing received</div>
            )}
            {damageToTeam === 0 && healAmount === 0 && (
              <div className="text-gray-500">No change</div>
            )}
          </div>
        </div>
      </div>

      {/* Action Summary */}
      <div className="bg-blue-50 border-2 border-blue-300 rounded-lg p-3 mb-4">
        <div className="text-xs font-semibold text-gray-600 mb-2 uppercase tracking-wide">What Happened This Round</div>

        {/* Team Actions */}
        <div className="mb-3">
          <div className="text-xs text-gray-500 mb-1">Team Actions:</div>
          <div className="flex justify-center gap-4">
            {actions.map((action, playerId) => {
              const entry = allPlayers[playerId];
              if (!entry) return null;
              const isCurrentPlayer = entry.type === "real" && entry.player?.id === currentPlayerId;

              return (
                <div key={playerId} className="flex flex-col items-center">
                  <div className="text-3xl mb-1">{actionIcons[action]}</div>
                  <div className="text-xs text-gray-600">
                    {isCurrentPlayer ? "YOU" : `P${playerId + 1}`}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Enemy Action */}
        <div className="border-t border-blue-200 pt-2">
          <div className="text-xs text-gray-500 mb-1">Enemy Action:</div>
          <div className="flex justify-center items-center gap-2">
            <div className="text-3xl">{enemyIntent === "WILL_ATTACK" ? "⚔️" : "😴"}</div>
            <div className="text-sm font-medium text-gray-700">
              {enemyIntent === "WILL_ATTACK" ? "Enemy attacked!" : "Enemy did nothing"}
            </div>
          </div>
        </div>
      </div>

      <div className="text-center text-gray-500 text-sm">
        {countdown !== null && countdown > 0 ? (
          <div className="text-lg font-bold text-blue-600">
            Next round in {countdown}...
          </div>
        ) : countdown === 0 ? (
          <div className="text-lg font-bold text-green-600">
            Starting now!
          </div>
        ) : (
          "Next round starting soon..."
        )}
      </div>
    </div>
  );
});

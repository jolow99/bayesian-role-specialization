import React from "react";

const actionIcons = {
  ATTACK: "⚔️",
  BLOCK: "🛡️",
  HEAL: "💚"
};

export const ResultsPanel = React.memo(function ResultsPanel({
  roundNumber,
  stageNumber,
  turnNumber,
  actions = [],
  allPlayers = [],
  currentPlayerId,
  enemyIntent,
  countdown
}) {
  return (
    <div className="animate-fadeIn">
      <div className="bg-gray-800 text-white rounded-lg px-4 py-3 text-center mb-4">
        <h3 className="text-xl font-bold">
          Stage {stageNumber} - Turn {turnNumber} Results
        </h3>
      </div>

      {/* Action Summary */}
      <div className="bg-blue-50 border-2 border-blue-300 rounded-lg p-4 mb-4">
        <div className="text-xs font-semibold text-gray-600 mb-2 uppercase tracking-wide">What Happened This Turn</div>

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
            Next turn in {countdown}...
          </div>
        ) : countdown === 0 ? (
          <div className="text-lg font-bold text-green-600">
            Starting now!
          </div>
        ) : (
          "Next turn starting soon..."
        )}
      </div>
    </div>
  );
});

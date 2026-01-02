import React from "react";
import { HealthBar } from "./HealthBar";

const actionIcons = {
  ATTACK: "⚔️",
  DEFEND: "🛡️",
  HEAL: "💚"
};

export const BattleField = React.memo(function BattleField({
  enemyHealth,
  maxEnemyHealth,
  teamHealth,
  maxHealth,
  enemyIntent,
  isRevealStage,
  showDamageAnimation,
  damageToEnemy,
  damageToTeam,
  healAmount,
  actions = [],
  allPlayers = [],
  currentPlayerId
}) {
  return (
    <div className="bg-gradient-to-b from-green-200 to-green-300 p-8 relative overflow-hidden" style={{ minHeight: '320px' }}>
      {/* Enemy Side (Top Right) */}
      <div className="absolute top-8 right-16 flex flex-col items-center">
        <div className="relative">
          {/* Enemy action icon (if reveal stage) */}
          {isRevealStage && (
            <div className="text-5xl mb-2 animate-bounce">
              {enemyIntent === "WILL_ATTACK" ? "⚔️" : "😴"}
            </div>
          )}
          <div className="text-9xl mb-4">👹</div>
          {/* Damage animation */}
          {isRevealStage && damageToEnemy > 0 && showDamageAnimation && (
            <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 text-3xl font-bold text-red-600 animate-bounce">
              -{damageToEnemy}
            </div>
          )}
        </div>
        {/* Enemy health bar below */}
        <div className="w-64">
          <HealthBar label="" current={enemyHealth} max={maxEnemyHealth} color="red" />
        </div>
      </div>

      {/* Team Side (Bottom Left) */}
      <div className="absolute bottom-8 left-16 flex flex-col items-center">
        <div className="flex items-end justify-center gap-6 mb-4">
          {/* Sort players: left teammate, YOU (center), right teammate */}
          {allPlayers
            .map((entry, playerId) => ({ entry, playerId }))
            .filter(({ entry }) => entry !== null)
            .sort((a, b) => {
              const aIsYou = a.entry.type === "real" && a.entry.player?.id === currentPlayerId;
              const bIsYou = b.entry.type === "real" && b.entry.player?.id === currentPlayerId;
              if (aIsYou) return 0; // YOU in middle
              if (bIsYou) return 0;
              // Others: maintain relative order
              return a.playerId - b.playerId;
            })
            .map(({ entry, playerId }, sortedIdx) => {
              const isCurrentPlayer = entry.type === "real" && entry.player?.id === currentPlayerId;
              const stats = entry.type === "real" ? entry.player.get("stats") : entry.bot.stats;
              const size = isCurrentPlayer ? "text-7xl" : "text-5xl";

              // Determine order: left, center (YOU), right
              let orderClass = '';
              if (isCurrentPlayer) {
                orderClass = 'order-2';
              } else if (sortedIdx === 0) {
                orderClass = 'order-1';
              } else {
                orderClass = 'order-3';
              }

              const maxStat = 6; // Stats sum to 6

              return (
                <div key={playerId} className={`flex flex-col items-center ${orderClass}`}>
                  {/* Action emoji (if reveal stage) */}
                  {isRevealStage && actions[playerId] && (
                    <div className="text-4xl mb-1 animate-bounce">
                      {actionIcons[actions[playerId]]}
                    </div>
                  )}
                  {/* Stats above player with bars */}
                  <div className="bg-white/90 rounded px-2 py-2 mb-2 border border-gray-400" style={{ width: '110px' }}>
                    <div className="space-y-1">
                      {/* STR */}
                      <div className="flex items-center gap-1">
                        <span className="text-xs font-semibold w-6">STR</span>
                        <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-red-500 transition-all"
                            style={{ width: `${(stats.STR / maxStat) * 100}%` }}
                          />
                        </div>
                        <span className="text-xs font-bold w-3 text-right">{stats.STR}</span>
                      </div>
                      {/* DEF */}
                      <div className="flex items-center gap-1">
                        <span className="text-xs font-semibold w-6">DEF</span>
                        <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-blue-500 transition-all"
                            style={{ width: `${(stats.DEF / maxStat) * 100}%` }}
                          />
                        </div>
                        <span className="text-xs font-bold w-3 text-right">{stats.DEF}</span>
                      </div>
                      {/* SUP */}
                      <div className="flex items-center gap-1">
                        <span className="text-xs font-semibold w-6">SUP</span>
                        <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-green-500 transition-all"
                            style={{ width: `${(stats.SUP / maxStat) * 100}%` }}
                          />
                        </div>
                        <span className="text-xs font-bold w-3 text-right">{stats.SUP}</span>
                      </div>
                    </div>
                  </div>
                  {/* Player sprite */}
                  <div className={size}>👤</div>
                  {/* Player label */}
                  <div className={`text-xs font-bold text-gray-700 mt-1 ${isCurrentPlayer ? 'text-sm' : ''}`}>
                    {isCurrentPlayer ? "YOU" : `P${playerId + 1}`}
                  </div>
                </div>
              );
            })}
        </div>
        {/* Team health bar below with damage/heal animations */}
        <div className="w-64 relative">
          <HealthBar label="" current={teamHealth} max={maxHealth} color="green" />
          {/* Damage/Heal animations */}
          {isRevealStage && showDamageAnimation && (
            <>
              {damageToTeam > 0 && (
                <div className="absolute -top-10 left-1/2 transform -translate-x-1/2 text-2xl font-bold text-orange-600 animate-bounce">
                  -{damageToTeam}
                </div>
              )}
              {healAmount > 0 && (
                <div className="absolute -top-10 right-0 text-2xl font-bold text-green-600 animate-bounce">
                  +{healAmount}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
});

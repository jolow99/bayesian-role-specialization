import React from "react";
import { HealthBar } from "./HealthBar";
import { ACTION_ICONS } from "../constants";

const actionIcons = ACTION_ICONS;

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
  currentPlayerGameId,
  previousEnemyHealth,
  previousTeamHealth,
  bossDamage,
  enemyAttackProbability
}) {
  return (
    <div className="bg-gradient-to-b from-green-200 to-green-300 p-6 relative h-full overflow-hidden" data-tutorial-id="battlefield">
      {/* Enemy Side (Top Right) */}
      <div className="absolute top-4 right-12 flex flex-col items-center" data-tutorial-id="enemy-section">
        {/* Boss stats above boss */}
        <div className="bg-white/90 rounded px-2 py-1 mb-2 border border-gray-400" style={{ width: '120px' }} data-tutorial-id="boss-stats">
          <div className="space-y-1">
            {/* Boss STR */}
            <div className="flex items-center gap-1">
              <span className="text-xs font-semibold w-10">STR</span>
              <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-red-500 transition-all"
                  style={{ width: `${Math.min(100, (bossDamage / 6) * 100)}%` }}
                />
              </div>
              <span className="text-xs font-bold w-3 text-right">{bossDamage}</span>
            </div>
            {/* Boss Attack Chance */}
            <div className="flex items-center gap-1">
              <span className="text-xs font-semibold w-10">ATK%</span>
              <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-orange-500 transition-all"
                  style={{ width: `${enemyAttackProbability * 100}%` }}
                />
              </div>
              <span className="text-xs font-bold w-6 text-right">{Math.round(enemyAttackProbability * 100)}%</span>
            </div>
          </div>
        </div>
        <div className="relative">
          <div className="text-8xl mb-2">👹</div>
        </div>
        {/* Enemy health bar below */}
        <div className="w-56">
          <HealthBar
            label=""
            current={enemyHealth}
            max={maxEnemyHealth}
            color="red"
            previousHealth={previousEnemyHealth}
            damageAmount={damageToEnemy}
            healAmount={0}
            showChange={isRevealStage}
          />
        </div>
      </div>

      {/* Team Side (Bottom Left) */}
      <div className="absolute bottom-4 left-12 flex flex-col items-center" data-tutorial-id="team-section">
        <div className="flex items-end justify-center gap-4 mb-3">
          {/* Sort players: left teammate, YOU (center), right teammate */}
          {allPlayers
            .map((entry, playerId) => ({ entry, playerId }))
            .filter(({ entry }) => entry !== null)
            .sort((a, b) => {
              const aIsYou = a.playerId === currentPlayerGameId;
              const bIsYou = b.playerId === currentPlayerGameId;
              // Put YOU in the middle by sorting to index 1
              if (aIsYou && !bIsYou) return -1; // a (YOU) comes before b
              if (!aIsYou && bIsYou) return 1;  // b (YOU) comes before a
              // For non-YOU players, maintain their relative order
              return a.playerId - b.playerId;
            })
            .map(({ entry, playerId }, sortedIdx) => {
              const isCurrentPlayer = playerId === currentPlayerGameId;
              // Handle both tutorial mode (direct stats) and game mode (stats via .get())
              const stats = entry.type === "real"
                ? (entry.player.round?.get ? entry.player.round.get("stats") : entry.player.stats)
                : entry.bot.stats;
              const size = isCurrentPlayer ? "text-6xl" : "text-4xl";

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
                  {/* Stats above player with bars */}
                  <div className="bg-white/90 rounded px-2 py-1 mb-1 border border-gray-400" style={{ width: '100px' }} data-tutorial-id="player-stats">
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
        {/* Team health bar below */}
        <div className="w-56 relative">
          <HealthBar
            label=""
            current={teamHealth}
            max={maxHealth}
            color="green"
            previousHealth={previousTeamHealth}
            damageAmount={damageToTeam}
            healAmount={healAmount}
            showChange={isRevealStage}
          />
        </div>
      </div>
    </div>
  );
});

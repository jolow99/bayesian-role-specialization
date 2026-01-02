import React from "react";
import { useGame, usePlayer, usePlayers } from "@empirica/core/player/classic/react";

export function ActionHistory({ maxRows }) {
  const game = useGame();
  const player = usePlayer();
  const players = usePlayers();

  // Get team action history from game
  const teamHistory = game.get("teamActionHistory") || [];

  const actionIcons = {
    ATTACK: "⚔️",
    DEFEND: "🛡️",
    HEAL: "💚"
  };

  // Show all history (no limit) or use maxRows
  const displayHistory = maxRows ? teamHistory.slice(-maxRows).reverse() : teamHistory.slice().reverse();

  if (displayHistory.length === 0) {
    return (
      <div className="text-xs text-gray-500 italic text-center py-4">
        No rounds completed yet...
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {displayHistory.map((entry, idx) => (
        <div
          key={idx}
          className="bg-gray-50 rounded-lg border border-gray-300 p-3"
        >
          {/* Round header with health info */}
          <div className="flex items-center justify-between mb-2">
            <span className="font-bold text-sm text-gray-800">Round {entry.round}</span>
            <div className="flex gap-3 text-xs">
              <span className="text-red-600 font-semibold">
                👹 {entry.enemyHealth}HP
              </span>
              <span className="text-green-600 font-semibold">
                👥 {entry.teamHealth}HP
              </span>
            </div>
          </div>

          {/* Player actions */}
          <div className="flex gap-2 justify-center">
            {entry.actions && entry.actions.map((playerAction, pidx) => (
              <div
                key={pidx}
                className="flex flex-col items-center bg-white rounded border border-gray-300 px-2 py-1"
              >
                <span className="text-2xl">{actionIcons[playerAction.action]}</span>
                <span className="text-xs text-gray-600">P{pidx + 1}</span>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

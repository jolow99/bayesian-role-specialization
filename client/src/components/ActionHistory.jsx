import React from "react";
import { useGame, usePlayer } from "@empirica/core/player/classic/react";

export function ActionHistory({ maxRows = 5 }) {
  const game = useGame();
  const player = usePlayer();

  // Get action history from player's stored history
  const playerHistory = player.get("actionHistory") || [];

  // Convert to display format
  const history = playerHistory.map(entry => ({
    round: entry.round,
    action: entry.action,
    enemyHealth: entry.enemyHealth,
    teamHealth: entry.teamHealth
  }));

  const actionIcons = {
    ATTACK: "⚔️",
    DEFEND: "🛡️",
    HEAL: "💚"
  };

  // Show only the most recent maxRows
  const recentHistory = history.slice(-maxRows).reverse();

  if (recentHistory.length === 0) {
    return (
      <div className="bg-gray-50 rounded-lg border border-gray-200 p-4">
        <h3 className="text-sm font-semibold text-gray-700 mb-2">Your Action History</h3>
        <p className="text-xs text-gray-500 italic">No actions yet...</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-50 rounded-lg border border-gray-200 p-4">
      <h3 className="text-sm font-semibold text-gray-700 mb-3">Your Action History</h3>
      <div className="space-y-2">
        {recentHistory.map((entry, idx) => (
          <div
            key={idx}
            className="bg-white rounded border border-gray-200 p-2 text-xs"
          >
            <div className="flex justify-between items-center">
              <span className="font-semibold text-gray-700">R{entry.round}: {actionIcons[entry.action] || entry.action}</span>
              <span className="text-gray-500 text-[10px]">
                E: {entry.enemyHealth} | T: {entry.teamHealth}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

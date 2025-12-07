import React from "react";
import { useGame, usePlayer, usePlayers } from "@empirica/core/player/classic/react";

export function ActionHistory({ maxRows = 5 }) {
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

  // Show only the most recent maxRows
  const recentHistory = teamHistory.slice(-maxRows).reverse();

  if (recentHistory.length === 0) {
    return (
      <div className="bg-gray-50 rounded-lg border border-gray-200 p-4">
        <h3 className="text-sm font-semibold text-gray-700 mb-2">Team Action History</h3>
        <p className="text-xs text-gray-500 italic">No actions yet...</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-50 rounded-lg border border-gray-200 p-4">
      <h3 className="text-sm font-semibold text-gray-700 mb-3">Team Action History</h3>
      <div className="space-y-2">
        {recentHistory.map((entry, idx) => (
          <div
            key={idx}
            className="bg-white rounded border border-gray-200 p-2"
          >
            <div className="mb-1">
              <span className="font-semibold text-gray-700 text-xs">Round {entry.round}</span>
            </div>
            <div className="flex gap-1">
              {entry.actions && entry.actions.map((playerAction, pidx) => (
                <div
                  key={pidx}
                  className="flex-1 text-center text-xs bg-gray-50 rounded px-1 py-0.5"
                  title={`Player ${pidx + 1}: ${playerAction.action}`}
                >
                  <span className="text-base">{actionIcons[playerAction.action]}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

import React, { useMemo } from "react";
import { useGame, usePlayer, usePlayers } from "@empirica/core/player/classic/react";
import { useTutorialContext } from "./tutorial/TutorialContext";

export function ActionHistory({ maxRows }) {
  // Check if in tutorial mode
  const { isTutorialMode, mockData } = useTutorialContext();

  // Use mock data if in tutorial mode, otherwise use Empirica hooks
  const game = isTutorialMode ? mockData.game : useGame();
  const player = isTutorialMode ? mockData.player : usePlayer();
  const players = isTutorialMode ? mockData.players : usePlayers();

  // Get team action history from game
  const teamHistory = isTutorialMode ? (mockData.teamHistory || []) : (game.get("teamActionHistory") || []);

  const actionIcons = {
    ATTACK: "⚔️",
    DEFEND: "🛡️",
    HEAL: "💚"
  };

  const roleNames = ["Fighter", "Tank", "Healer"];

  // Group turns by round
  const roundHistory = useMemo(() => {
    const grouped = {};

    teamHistory.forEach(entry => {
      if (!grouped[entry.round]) {
        grouped[entry.round] = [];
      }
      grouped[entry.round].push(entry);
    });

    // Convert to array and sort by round number (ascending for chronological order)
    return Object.entries(grouped)
      .map(([round, turns]) => ({
        round: parseInt(round),
        turns: turns.sort((a, b) => a.turn - b.turn) // Sort turns within round ascending (Turn 1, then Turn 2)
      }))
      .sort((a, b) => a.round - b.round); // Sort rounds ascending (Round 1, Round 2, Round 3...)
  }, [teamHistory]);

  // Get player's action history to find their role for each round
  const playerActionHistory = player.get("actionHistory") || [];

  if (roundHistory.length === 0) {
    return (
      <div className="text-xs text-gray-500 italic text-center py-4">
        No rounds completed yet...
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {roundHistory.map((roundEntry, idx) => {
        // Find player's role for this round from their action history
        const playerRoundData = playerActionHistory.find(h => h.round === roundEntry.round);
        const playerRole = playerRoundData?.role;

        return (
          <div
            key={idx}
            className="bg-gray-50 rounded-lg border border-gray-300 p-3"
          >
            {/* Round header with player role */}
            <div className="mb-2 flex items-center justify-between">
              <span className="font-bold text-sm text-gray-800">Round {roundEntry.round}</span>
              {playerRole !== null && playerRole !== undefined && (
                <div className="text-xs text-blue-600 font-semibold bg-blue-50 px-2 py-1 rounded">
                  Your role: {roleNames[playerRole]}
                </div>
              )}
            </div>

            {/* Turns within the round */}
            <div className="space-y-2">
              {roundEntry.turns.map((turn, turnIdx) => (
                <div
                  key={turnIdx}
                  className="bg-white rounded border border-gray-200 p-2"
                  data-tutorial-id={`battle-history-r${roundEntry.round}t${turn.turn}`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="text-xs text-gray-600 font-semibold">
                      Turn {turn.turn}
                      {turn.enemyIntent && (
                        <span className={`ml-2 ${turn.enemyIntent === 'WILL_ATTACK' ? 'text-red-600' : 'text-gray-500'}`}>
                          ({turn.enemyIntent === 'WILL_ATTACK' ? 'Enemy Attacked' : 'Enemy Rested'})
                        </span>
                      )}
                    </div>
                    <div className="flex gap-3 text-xs">
                      <span className="text-red-600 font-semibold">
                        👹 {turn.enemyHealth}HP
                      </span>
                      <span className="text-green-600 font-semibold">
                        👥 {turn.teamHealth}HP
                      </span>
                    </div>
                  </div>
                  <div className="flex gap-2 justify-center">
                    {turn.actions && turn.actions.map((playerAction, pidx) => (
                      <div
                        key={pidx}
                        className="flex flex-col items-center bg-gray-50 rounded border border-gray-300 px-2 py-1"
                      >
                        <span className="text-2xl">{actionIcons[playerAction.action]}</span>
                        <span className="text-xs text-gray-600">P{pidx + 1}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

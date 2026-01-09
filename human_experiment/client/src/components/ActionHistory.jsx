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

  // Group turns by stage (each stage has 2 turns)
  const stageHistory = useMemo(() => {
    const grouped = {};

    teamHistory.forEach(entry => {
      if (!grouped[entry.stage]) {
        grouped[entry.stage] = [];
      }
      grouped[entry.stage].push(entry);
    });

    // Convert to array and sort by stage number (ascending for chronological order)
    return Object.entries(grouped)
      .map(([stage, turns]) => ({
        stage: parseInt(stage),
        turns: turns.sort((a, b) => a.turn - b.turn) // Sort turns within stage ascending (Turn 1, then Turn 2)
      }))
      .sort((a, b) => a.stage - b.stage); // Sort stages ascending (Stage 1, Stage 2, Stage 3...)
  }, [teamHistory]);

  // Get player's action history to find their role for each stage
  const playerActionHistory = player.get("actionHistory") || [];

  // Map role names (stored as "FIGHTER", "TANK", "HEALER") to display names
  const roleNameMap = {
    "FIGHTER": "Fighter",
    "TANK": "Tank",
    "HEALER": "Healer"
  };

  if (stageHistory.length === 0) {
    return (
      <div className="text-xs text-gray-500 italic text-center py-4">
        No stages completed yet...
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {stageHistory.map((stageEntry, idx) => {
        // Find player's role for this stage from their action history
        const playerStageData = playerActionHistory.find(h => h.stage === stageEntry.stage);
        const playerRole = playerStageData?.role; // This is a string like "FIGHTER"
        const displayRole = playerRole ? roleNameMap[playerRole] : null;

        return (
          <div
            key={idx}
            className="bg-gray-50 rounded-lg border border-gray-300 p-3"
          >
            {/* Stage header with player role */}
            <div className="mb-2 flex items-center justify-between">
              <span className="font-bold text-sm text-gray-800">Stage {stageEntry.stage}</span>
              {displayRole && (
                <div className="text-xs text-blue-600 font-semibold bg-blue-50 px-2 py-1 rounded">
                  Your role: {displayRole}
                </div>
              )}
            </div>

            {/* Turns within the stage */}
            <div className="space-y-2">
              {stageEntry.turns.map((turn, turnIdx) => (
                <div
                  key={turnIdx}
                  className="bg-white rounded border border-gray-200 p-2"
                  data-tutorial-id={`battle-history-s${stageEntry.stage}t${turn.turn}`}
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

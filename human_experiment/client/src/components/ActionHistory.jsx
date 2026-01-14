import React, { useMemo } from "react";
import { useGame, usePlayer, usePlayers, useRound } from "@empirica/core/player/classic/react";
import { useTutorialContext } from "./tutorial/TutorialContext";
import { ACTION_ICONS, ROLE_LABELS } from "../constants";

export function ActionHistory({ currentStageView = null, currentTurnView = 0 }) {
  // Check if in tutorial mode
  const { isTutorialMode, mockData } = useTutorialContext();

  // Use mock data if in tutorial mode, otherwise use Empirica hooks
  const game = isTutorialMode ? mockData.game : useGame();
  const player = isTutorialMode ? mockData.player : usePlayer();
  const players = isTutorialMode ? mockData.players : usePlayers();
  const round = isTutorialMode ? mockData.round : useRound();

  // Get current round number
  const currentRound = round?.get("roundNumber");
  const currentRoundStageNumber = round?.get("stageNumber") || 0;

  const actionIcons = ACTION_ICONS;
  const roleNames = ROLE_LABELS;

  // Build stage history from per-stage turn data stored on the round
  // This is synchronized with the frontend's turn-by-turn display
  const stageHistory = useMemo(() => {
    if (!round || !currentRound) return [];

    const stages = [];

    // Iterate through stages up to currentRoundStageNumber (which tracks completed stages)
    for (let stageNum = 1; stageNum <= currentRoundStageNumber; stageNum++) {
      const stageTurns = round.get(`stage${stageNum}Turns`);

      if (stageTurns && stageTurns.length > 0) {
        // Filter turns based on current viewing state
        let turnsToShow = stageTurns;

        // If we're currently viewing this stage, only show turns up to currentTurnView
        if (currentStageView === stageNum && currentTurnView > 0) {
          turnsToShow = stageTurns.slice(0, currentTurnView);
        }
        // If this is a future stage we haven't started viewing yet, don't show it
        else if (currentStageView !== null && stageNum > currentStageView) {
          continue;
        }

        // Only add stage if there are turns to show
        if (turnsToShow.length > 0) {
          // Convert turn data to match the expected format
          const turns = turnsToShow.map(turn => ({
            round: currentRound,
            stage: stageNum,
            turn: turn.turnNumber,
            actions: turn.actions.map((action, idx) => ({ playerId: idx, action })),
            enemyHealth: turn.newEnemyHealth,
            teamHealth: turn.newTeamHealth,
            enemyIntent: turn.enemyIntent
          }));

          stages.push({
            stage: stageNum,
            turns: turns
          });
        }
      }
    }

    return stages;
  }, [round, currentRound, currentRoundStageNumber, currentStageView, currentTurnView]);

  // Get player's action history to find their role for each stage
  const playerActionHistory = player.get("actionHistory") || [];

  // Get current player's playerId (their slot position: 0, 1, or 2)
  const currentPlayerPlayerId = player.round?.get ? player.round.get("playerId") : player.playerId;

  // Map role names (stored as "FIGHTER", "TANK", "MEDIC") to display names
  const roleNameMap = {
    "FIGHTER": ROLE_LABELS[0],
    "TANK": ROLE_LABELS[1],
    "MEDIC": ROLE_LABELS[2]
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
                    {turn.actions && turn.actions.map((playerAction, pidx) => {
                      const isCurrentPlayer = pidx === currentPlayerPlayerId;
                      return (
                        <div
                          key={pidx}
                          className="flex flex-col items-center bg-gray-50 rounded border border-gray-300 px-2 py-1"
                        >
                          <span className="text-2xl">{actionIcons[playerAction.action]}</span>
                          <span className={`text-xs ${isCurrentPlayer ? 'text-blue-600 font-bold' : 'text-gray-600'}`}>
                            {isCurrentPlayer ? "YOU" : `P${pidx + 1}`}
                          </span>
                        </div>
                      );
                    })}
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

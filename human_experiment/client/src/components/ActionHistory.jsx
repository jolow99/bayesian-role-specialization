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

  // Derive isBotRound from shuffledRoundOrder (same logic as server)
  // This is more reliable than round.get("isBotRound") due to Empirica sync timing
  const shuffledRoundOrder = game?.get("shuffledRoundOrder") || [];
  const roundSlot = currentRound ? shuffledRoundOrder[currentRound - 1] : null;
  const isBotRound = roundSlot?.type === "bot";

  // Get round config for max health values
  // For bot rounds, use player-specific config; for human rounds, use shared round config
  const roundConfig = isBotRound
    ? (player?.round?.get ? player.round.get("playerRoundConfig") : null)
    : (currentRound ? game?.get(`round${currentRound}Config`) : null);
  const maxTeamHealth = roundConfig?.maxTeamHealth;
  const maxEnemyHealth = roundConfig?.maxEnemyHealth;

  const actionIcons = ACTION_ICONS;
  const roleNames = ROLE_LABELS;

  // // Read all stage turns data outside useMemo to ensure Empirica's reactivity tracks changes
  // // For bot rounds, each player has their own turn data; for human rounds, it's shared on round
  // const allStageTurnsData = [];
  // for (let stageNum = 1; stageNum <= currentRoundStageNumber; stageNum++) {
  //   const stageTurns = isBotRound
  //     ? player?.round?.get(`stage${stageNum}Turns`)
  //     : round?.get(`stage${stageNum}Turns`);
  //   allStageTurnsData.push(stageTurns || null);
  // }

  // Build stage history from per-stage turn data stored on the round (or player.round for bot rounds)
  // This is synchronized with the frontend's turn-by-turn display
  const stageHistory = useMemo(() => {
    if (!round || !currentRound) return [];

    const stages = [];

    // Iterate through stages up to currentRoundStageNumber (which tracks completed stages)
    for (let stageNum = 1; stageNum <= currentRoundStageNumber; stageNum++) {
      // In bot rounds, read from player-specific state; in human rounds, from shared round state
      const stageTurns = isBotRound && player?.round?.get
        ? player.round.get(`stage${stageNum}Turns`)
        : round.get(`stage${stageNum}Turns`);

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
            enemyIntent: turn.enemyIntent,
            // Calculate health changes for display
            teamHealthChange: turn.newTeamHealth - turn.previousTeamHealth,
            enemyHealthChange: turn.newEnemyHealth - turn.previousEnemyHealth
          }));

          stages.push({
            stage: stageNum,
            turns: turns
          });
        }
      }
    }

    return stages;
  }, [round, currentRound, currentRoundStageNumber, currentStageView, currentTurnView, isBotRound]);

  // Get player's action history to find their role for each stage
  const playerActionHistory = player.get("actionHistory") || [];

  // Get current player's permanent gamePlayerId (their slot position: 0, 1, or 2)
  const currentPlayerPlayerId = player?.get ? player.get("gamePlayerId") : player.playerId;

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
      {/* Starting state - shows initial health before any stages */}
      {maxTeamHealth && maxEnemyHealth && (
        <div className="bg-gray-50 rounded-lg border border-gray-300 p-3">
          <div className="mb-2">
            <span className="font-bold text-sm text-gray-800">Starting State</span>
          </div>
          <div className="flex gap-2">
            {/* Team starting health */}
            <div className="flex-1 bg-green-50 rounded p-1.5 border border-green-200">
              <div className="text-xs font-semibold text-green-700 flex items-center justify-center gap-1">
                <span>👥</span>
                <span>{maxTeamHealth}HP</span>
              </div>
            </div>
            {/* Enemy starting health */}
            <div className="flex-1 bg-red-50 rounded p-1.5 border border-red-200">
              <div className="text-xs font-semibold text-red-700 flex items-center justify-center gap-1">
                <span>👹</span>
                <span>{maxEnemyHealth}HP</span>
              </div>
            </div>
          </div>
        </div>
      )}

      {stageHistory.map((stageEntry, idx) => {
        // Find player's role for this stage from their action history
        // Must match both round AND stage since actionHistory accumulates across all rounds
        const playerStageData = playerActionHistory.find(h => h.round === currentRound && h.stage === stageEntry.stage);
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
                  {/* Turn header */}
                  <div className="text-xs text-gray-600 font-semibold text-center mb-2 border-b border-gray-100 pb-1">
                    Turn {turn.turn}
                  </div>

                  {/* Two-column layout: Team (left) | Enemy (right) */}
                  <div className="flex gap-2">
                    {/* Left: Team side */}
                    <div className="flex-1 bg-green-50 rounded p-1.5 border border-green-200">
                      {/* Team HP with change */}
                      <div className="text-xs font-semibold text-green-700 mb-1 flex items-center justify-center gap-1">
                        <span>👥</span>
                        <span>{turn.teamHealth}HP</span>
                        {turn.teamHealthChange !== 0 && (
                          <span className={turn.teamHealthChange > 0 ? 'text-green-600' : 'text-red-500'}>
                            ({turn.teamHealthChange > 0 ? '+' : ''}{turn.teamHealthChange})
                          </span>
                        )}
                      </div>
                      {/* Team actions */}
                      <div className="flex gap-1 justify-center">
                        {turn.actions && turn.actions.map((playerAction, pidx) => {
                          const isCurrentPlayer = pidx === currentPlayerPlayerId;
                          return (
                            <div
                              key={pidx}
                              className="flex flex-col items-center"
                            >
                              <span className="text-lg">{actionIcons[playerAction.action]}</span>
                              <span className={`text-xs ${isCurrentPlayer ? 'text-blue-600 font-bold' : 'text-gray-500'}`}>
                                {isCurrentPlayer ? "YOU" : `P${pidx + 1}`}
                              </span>
                            </div>
                          );
                        })}
                      </div>
                    </div>

                    {/* Right: Enemy side */}
                    <div className="flex-1 bg-red-50 rounded p-1.5 border border-red-200">
                      {/* Enemy HP with change */}
                      <div className="text-xs font-semibold text-red-700 mb-1 flex items-center justify-center gap-1">
                        <span>👹</span>
                        <span>{turn.enemyHealth}HP</span>
                        {turn.enemyHealthChange !== 0 && (
                          <span className="text-red-500">
                            ({turn.enemyHealthChange})
                          </span>
                        )}
                      </div>
                      {/* Enemy action */}
                      <div className="flex flex-col items-center justify-center">
                        <span className="text-2xl">
                          {turn.enemyIntent === 'WILL_ATTACK' ? '⚔️' : '😴'}
                        </span>
                      </div>
                    </div>
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

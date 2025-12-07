import React, { useState } from "react";
import { usePlayer, useGame, useRound, usePlayers } from "@empirica/core/player/classic/react";
import { ActionButton } from "../components/ActionButton";
import { HealthBar } from "../components/HealthBar";
import { PlayerStats } from "../components/PlayerStats";
import { ActionHistory } from "../components/ActionHistory";

const ACTIONS = { ATTACK: 0, DEFEND: 1, HEAL: 2 };

export function ActionSelection() {
  const player = usePlayer();
  const players = usePlayers();
  const game = useGame();
  const round = useRound();

  const [selectedAction, setSelectedAction] = useState(null);

  const submitted = player.round.get("submitted");
  const stats = player.get("stats");
  const enemyHealth = round.get("enemyHealth") || 10;
  const teamHealth = round.get("teamHealth") || 10;
  const enemyIntent = round.get("enemyIntent");
  const roundNumber = round.get("roundNumber");
  const maxRounds = game.get("maxRounds");
  const maxHealth = game.get("maxHealth") || 10;

  // Always show stats (no priorType factor anymore)
  const showStats = true;

  const handleActionSelect = (action) => {
    if (!submitted) {
      setSelectedAction(action);
    }
  };

  const handleSubmit = () => {
    if (selectedAction !== null) {
      player.round.set("action", selectedAction);
      player.round.set("submitted", true);
    }
  };

  if (submitted) {
    return (
      <div className="flex flex-col items-center justify-center h-full">
        <div className="text-center text-gray-500 mb-4">
          <div className="text-2xl mb-2">⏳</div>
          <div className="text-lg font-semibold">Waiting for other players...</div>
          <div className="text-sm mt-2">You selected: {["Attack", "Defend", "Heal"][selectedAction]}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="w-full max-w-6xl mx-auto p-6">
      <div className="grid grid-cols-3 gap-6 mb-6">
        {/* Left Column: Enemy Info */}
        <div className="space-y-4">
          <div className="bg-red-50 border-2 border-red-300 rounded-lg p-4">
            <h2 className="text-xl font-bold text-red-700 mb-4 text-center">Enemy</h2>
            <HealthBar label="Enemy Health" current={enemyHealth} max={maxHealth} color="red" />
            <div className="mt-4 p-3 bg-white rounded border border-red-200">
              <div className="text-sm font-semibold text-gray-700 mb-1">Enemy Intent:</div>
              <div className={`text-lg font-bold ${enemyIntent === "WILL_ATTACK" ? "text-red-600" : "text-gray-500"}`}>
                {enemyIntent === "WILL_ATTACK" ? "⚠️ Will Attack!" : "😴 Resting"}
              </div>
            </div>
          </div>
        </div>

        {/* Middle Column: Action Selection */}
        <div className="space-y-4">
          <div className="bg-white border-2 border-gray-300 rounded-lg p-4">
            <div className="text-center mb-4">
              <h2 className="text-2xl font-bold text-gray-800">Round {roundNumber}/{maxRounds}</h2>
              <p className="text-sm text-gray-600 mt-1">Choose your action</p>
            </div>

            <div className="grid grid-cols-3 gap-3 mb-4">
              <ActionButton
                action="ATTACK"
                selected={selectedAction === ACTIONS.ATTACK}
                onClick={() => handleActionSelect(ACTIONS.ATTACK)}
                disabled={submitted}
              />
              <ActionButton
                action="DEFEND"
                selected={selectedAction === ACTIONS.DEFEND}
                onClick={() => handleActionSelect(ACTIONS.DEFEND)}
                disabled={submitted}
              />
              <ActionButton
                action="HEAL"
                selected={selectedAction === ACTIONS.HEAL}
                onClick={() => handleActionSelect(ACTIONS.HEAL)}
                disabled={submitted}
              />
            </div>

            <button
              onClick={handleSubmit}
              disabled={selectedAction === null}
              className={`w-full py-3 px-4 rounded-lg font-bold text-white transition-all ${
                selectedAction === null
                  ? "bg-gray-300 cursor-not-allowed"
                  : "bg-empirica-500 hover:bg-empirica-600 shadow-lg hover:shadow-xl"
              }`}
            >
              {selectedAction === null ? "Select an action first" : "Confirm Action"}
            </button>
          </div>

          <HealthBar label="Team Health" current={teamHealth} max={maxHealth} color="green" />
        </div>

        {/* Right Column: Stats and History */}
        <div className="space-y-4">
          {/* All Players' Stats */}
          {showStats && (
            <div className="bg-gray-50 rounded-lg border border-gray-200 p-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-3">Team Stats</h3>
              <div className="space-y-3">
                {players.map((p, idx) => (
                  <PlayerStats
                    key={p.id}
                    stats={p.get("stats")}
                    showStats={true}
                    playerName={p.id === player.id ? "You (P" + (idx + 1) + ")" : "Player " + (idx + 1)}
                  />
                ))}
              </div>
            </div>
          )}
          <ActionHistory maxRows={4} />
        </div>
      </div>
    </div>
  );
}

import React, { useState, useEffect } from "react";
import { usePlayer, useGame, useRound, usePlayers } from "@empirica/core/player/classic/react";
import { RoleButton } from "../components/RoleButton";
import { HealthBar } from "../components/HealthBar";
import { PlayerStats } from "../components/PlayerStats";
import { ActionHistory } from "../components/ActionHistory";

const ROLES = { FIGHTER: 0, TANK: 1, HEALER: 2 };

export function ActionSelection() {
  const player = usePlayer();
  const players = usePlayers();
  const game = useGame();
  const round = useRound();

  const [selectedRole, setSelectedRole] = useState(null);

  const submitted = player.stage.get("submit");
  const enemyHealth = game.get("enemyHealth") || 10;
  const teamHealth = game.get("teamHealth") || 10;
  const enemyIntent = round.get("enemyIntent");
  const roundNumber = round.get("roundNumber");
  const maxRounds = game.get("maxRounds");
  const maxHealth = game.get("maxHealth") || 10;

  // Role commitment state
  const currentRole = player.get("currentRole");
  const roleEndRound = player.get("roleEndRound");
  const isRoleCommitted = currentRole !== null;
  const roundsRemaining = isRoleCommitted ? (roleEndRound - roundNumber + 1) : 0;

  // Auto-submit when role is already committed
  useEffect(() => {
    if (isRoleCommitted && !submitted) {
      player.stage.set("submit", true);
    }
  }, [isRoleCommitted, submitted, player]);

  const handleRoleSelect = (role) => {
    if (!submitted && !isRoleCommitted) {
      setSelectedRole(role);
    }
  };

  const handleSubmit = () => {
    if (isRoleCommitted) {
      player.stage.set("submit", true);
    } else if (selectedRole !== null) {
      player.round.set("selectedRole", selectedRole);
      player.stage.set("submit", true);
    }
  };

  if (submitted) {
    const roleNames = ["Fighter", "Tank", "Healer"];
    const displayRole = isRoleCommitted ? roleNames[currentRole] : roleNames[selectedRole];

    return (
      <div className="flex flex-col items-center justify-center h-full bg-gradient-to-b from-blue-400 to-blue-600">
        <div className="text-center text-white mb-4">
          <div className="text-2xl mb-2">⏳</div>
          <div className="text-lg font-semibold">Waiting for other players...</div>
          <div className="text-sm mt-2">
            Your role: {displayRole}
            {isRoleCommitted && ` (${roundsRemaining} rounds remaining)`}
          </div>
        </div>
      </div>
    );
  }

  // Find current player index
  const currentPlayerIndex = players.findIndex(p => p.id === player.id);

  return (
    <div className="w-full h-full bg-gradient-to-b from-blue-400 to-blue-600 flex items-center justify-center p-4">
      <div className="w-full max-w-6xl">
        {/* Battle Screen */}
        <div className="bg-white rounded-lg shadow-2xl overflow-hidden border-4 border-gray-800">
          {/* Round Header */}
          <div className="bg-gray-800 text-white text-center py-3">
            <h1 className="text-2xl font-bold">Round {roundNumber}/{maxRounds}</h1>
            {isRoleCommitted && (
              <p className="text-sm text-yellow-300">
                Role: {["Fighter", "Tank", "Healer"][currentRole]} ({roundsRemaining} rounds left)
              </p>
            )}
          </div>

          {/* Battle Field */}
          <div className="bg-gradient-to-b from-green-200 to-green-300 p-8 relative" style={{ minHeight: '400px' }}>
            {/* Enemy Side (Top Right) */}
            <div className="absolute top-8 right-16 flex flex-col items-center">
              <div className="text-9xl mb-4">👹</div>
              {/* Enemy health bar below */}
              <div className="w-64">
                <HealthBar label="" current={enemyHealth} max={maxHealth} color="red" />
              </div>
            </div>

            {/* Team Side (Bottom Left) */}
            <div className="absolute bottom-8 left-16 flex flex-col items-center">
              <div className="flex items-end justify-center gap-6 mb-4">
                {/* Sort players: left teammate, YOU (center), right teammate */}
                {players
                  .map((p, idx) => ({ player: p, originalIdx: idx }))
                  .sort((a, b) => {
                    const aIsYou = a.player.id === player.id;
                    const bIsYou = b.player.id === player.id;
                    if (aIsYou) return 0; // YOU in middle
                    if (bIsYou) return 0;
                    // Others: maintain relative order
                    return a.originalIdx - b.originalIdx;
                  })
                  .map(({ player: p, originalIdx: idx }, sortedIdx) => {
                    const stats = p.get("stats");
                    const isCurrentPlayer = p.id === player.id;
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

                    return (
                      <div key={p.id} className={`flex flex-col items-center ${orderClass}`}>
                        {/* Stats above player */}
                        <div className="bg-white/90 rounded px-2 py-1 mb-2 text-xs font-mono whitespace-nowrap border border-gray-400">
                          {Math.round(stats.STR * 100)} STR / {Math.round(stats.DEF * 100)} DEF / {Math.round(stats.SUP * 100)} SUP
                        </div>
                        {/* Player sprite */}
                        <div className={size}>👤</div>
                        {/* Player label */}
                        <div className={`text-xs font-bold text-gray-700 mt-1 ${isCurrentPlayer ? 'text-sm' : ''}`}>
                          {isCurrentPlayer ? "YOU" : `P${idx + 1}`}
                        </div>
                      </div>
                    );
                  })}
              </div>
              {/* Team health bar below */}
              <div className="w-64">
                <HealthBar label="" current={teamHealth} max={maxHealth} color="green" />
              </div>
            </div>
          </div>

          {/* Action Menu */}
          <div className="bg-white border-t-4 border-gray-700">
            <div className="p-6">
              <div className="bg-gray-800 text-white rounded-t-lg px-4 py-2 text-sm font-bold">
                {isRoleCommitted
                  ? "Your role is locked for this round"
                  : "What role will you play?"}
              </div>
              <div className="bg-gray-100 rounded-b-lg border-2 border-gray-800 border-t-0 p-4">
                <div className="grid grid-cols-3 gap-3 mb-3">
                  <RoleButton
                    role="FIGHTER"
                    selected={selectedRole === ROLES.FIGHTER}
                    onClick={() => handleRoleSelect(ROLES.FIGHTER)}
                    disabled={submitted}
                    locked={isRoleCommitted && currentRole === ROLES.FIGHTER}
                  />
                  <RoleButton
                    role="TANK"
                    selected={selectedRole === ROLES.TANK}
                    onClick={() => handleRoleSelect(ROLES.TANK)}
                    disabled={submitted}
                    locked={isRoleCommitted && currentRole === ROLES.TANK}
                  />
                  <RoleButton
                    role="HEALER"
                    selected={selectedRole === ROLES.HEALER}
                    onClick={() => handleRoleSelect(ROLES.HEALER)}
                    disabled={submitted}
                    locked={isRoleCommitted && currentRole === ROLES.HEALER}
                  />
                </div>

                <button
                  onClick={handleSubmit}
                  disabled={!isRoleCommitted && selectedRole === null}
                  className={`w-full py-3 px-4 rounded-lg font-bold text-white transition-all ${
                    (!isRoleCommitted && selectedRole === null)
                      ? "bg-gray-400 cursor-not-allowed"
                      : "bg-blue-600 hover:bg-blue-700 shadow-lg"
                  }`}
                >
                  {isRoleCommitted
                    ? "▶ CONTINUE"
                    : selectedRole === null
                      ? "Select a role"
                      : "✓ CONFIRM ROLE (3 rounds)"}
                </button>
              </div>
            </div>
          </div>

          {/* Battle History */}
          <div className="bg-gray-50 border-t-2 border-gray-300 p-4">
            <div className="bg-white rounded-lg border-2 border-gray-400 p-4">
              <h3 className="text-lg font-bold text-gray-800 mb-3 flex items-center gap-2">
                📜 Battle History
              </h3>
              <ActionHistory />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

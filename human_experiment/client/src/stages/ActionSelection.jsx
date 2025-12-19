import React, { useState, useEffect } from "react";
import { usePlayer, useGame, useRound, usePlayers, useStage } from "@empirica/core/player/classic/react";
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
  const stage = useStage();

  const [selectedRole, setSelectedRole] = useState(null);
  const [showDamageAnimation, setShowDamageAnimation] = useState(false);
  const [countdown, setCountdown] = useState(null);

  const submitted = player.stage.get("submit");
  const enemyHealth = game.get("enemyHealth") || 10;
  const teamHealth = game.get("teamHealth") || 10;
  const enemyIntent = round.get("enemyIntent");
  const roundNumber = round.get("roundNumber");
  const maxRounds = game.get("maxRounds");
  const maxHealth = game.get("maxHealth") || 10;
  const currentStage = stage.get("name");
  const isRevealStage = currentStage === "Reveal";
  const actions = round.get("actions") || [];

  console.log(`[Client] Round ${roundNumber}, Stage: ${currentStage}, isRevealStage: ${isRevealStage}, submitted: ${submitted}`);
  const damageToEnemy = round.get("damageToEnemy") || 0;
  const damageToTeam = round.get("damageToTeam") || 0;
  const healAmount = round.get("healAmount") || 0;
  const previousEnemyHealth = round.get("previousEnemyHealth") || enemyHealth;
  const previousTeamHealth = round.get("previousTeamHealth") || teamHealth;

  const actionIcons = {
    ATTACK: "⚔️",
    DEFEND: "🛡️",
    HEAL: "💚"
  };

  // Role commitment state
  const currentRole = player.get("currentRole");
  const roleEndRound = player.get("roleEndRound");
  const isRoleCommitted = currentRole !== null;
  const roundsRemaining = isRoleCommitted ? (roleEndRound - roundNumber + 1) : 0;

  // Auto-submit based on stage and role commitment
  useEffect(() => {
    console.log(`[Auto-submit effect] isRoleCommitted: ${isRoleCommitted}, submitted: ${submitted}, isRevealStage: ${isRevealStage}`);
    if (submitted || isRevealStage) return;

    // Auto-submit when role is committed (instant to avoid UI flash)
    if (isRoleCommitted) {
      console.log(`[Auto-submit] Setting submit=true immediately`);
      const timer = setTimeout(() => {
        console.log(`[Auto-submit] NOW setting submit=true`);
        player.stage.set("submit", true);
      }, 0); // Instant submit to prevent UI flash
      return () => clearTimeout(timer);
    }
  }, [isRoleCommitted, submitted, isRevealStage, player]);

  // Trigger damage animation during reveal
  useEffect(() => {
    if (isRevealStage) {
      setShowDamageAnimation(true);
      const timer = setTimeout(() => setShowDamageAnimation(false), 12000); // Extended to 12 seconds for 15-second reveal
      return () => clearTimeout(timer);
    }
  }, [isRevealStage, roundNumber]);

  // Auto-submit after reveal stage duration (15 seconds)
  useEffect(() => {
    if (isRevealStage && !submitted) {
      console.log(`[Reveal] Auto-submitting after 15 seconds`);
      const timer = setTimeout(() => {
        console.log(`[Reveal] NOW submitting`);
        player.stage.set("submit", true);
      }, 15000); // Submit after 15 seconds
      return () => clearTimeout(timer);
    }
  }, [isRevealStage, submitted, player]);

  // Countdown timer for the last 5 seconds of reveal
  useEffect(() => {
    if (isRevealStage) {
      // Start countdown at 10 seconds (showing countdown for last 5 seconds)
      const countdownStart = setTimeout(() => {
        setCountdown(5);
      }, 10000);

      return () => clearTimeout(countdownStart);
    } else {
      setCountdown(null);
    }
  }, [isRevealStage, roundNumber]);

  // Update countdown every second
  useEffect(() => {
    if (countdown !== null && countdown > 0) {
      const timer = setTimeout(() => {
        setCountdown(countdown - 1);
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [countdown]);

  const handleRoleSelect = (role) => {
    if (!submitted && !isRoleCommitted) {
      setSelectedRole(role);
    }
  };

  const handleSubmit = () => {
    if (!submitted) {
      if (!isRoleCommitted && selectedRole !== null) {
        // New role selection
        player.round.set("selectedRole", selectedRole);
      }
      // Always submit (whether role is committed or newly selected)
      player.stage.set("submit", true);
    }
  };

  // Find current player index
  const currentPlayerIndex = players.findIndex(p => p.id === player.id);

  return (
    <div className="w-full h-full bg-gradient-to-b from-blue-400 to-blue-600 flex items-center justify-center p-4 transition-all duration-300">
      <div className="w-full max-w-6xl transition-all duration-300">
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
              <div className="relative">
                {/* Enemy action icon (if reveal stage) */}
                {isRevealStage && (
                  <div className="text-5xl mb-2 animate-bounce">
                    {enemyIntent === "WILL_ATTACK" ? "⚔️" : "😴"}
                  </div>
                )}
                <div className="text-9xl mb-4">👹</div>
                {/* Damage animation */}
                {isRevealStage && damageToEnemy > 0 && showDamageAnimation && (
                  <div className="absolute -top-8 left-1/2 transform -translate-x-1/2 text-3xl font-bold text-red-600 animate-bounce">
                    -{damageToEnemy}
                  </div>
                )}
              </div>
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
                        {/* Action emoji (if reveal stage) */}
                        {isRevealStage && actions[idx] && (
                          <div className="text-4xl mb-1 animate-bounce">
                            {actionIcons[actions[idx]]}
                          </div>
                        )}
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
              {/* Team health bar below with damage/heal animations */}
              <div className="w-64 relative">
                <HealthBar label="" current={teamHealth} max={maxHealth} color="green" />
                {/* Damage/Heal animations */}
                {isRevealStage && showDamageAnimation && (
                  <>
                    {damageToTeam > 0 && (
                      <div className="absolute -top-10 left-1/2 transform -translate-x-1/2 text-2xl font-bold text-orange-600 animate-bounce">
                        -{damageToTeam}
                      </div>
                    )}
                    {healAmount > 0 && (
                      <div className="absolute -top-10 right-0 text-2xl font-bold text-green-600 animate-bounce">
                        +{healAmount}
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Battle Results (during Reveal) or Action Menu (during Action Selection) */}
          <div className="bg-white border-t-4 border-gray-700">
            <div className="p-6">
              {submitted && !isRevealStage ? (
                /* Waiting for other players */
                <div className="text-center py-12">
                  <div className="text-6xl mb-4">⏳</div>
                  <div className="text-2xl font-bold text-gray-700 mb-2">Waiting for other players...</div>
                  <div className="text-gray-500">
                    {isRoleCommitted
                      ? `Your role: ${["Fighter", "Tank", "Healer"][currentRole]} (${roundsRemaining} rounds remaining)`
                      : `Your role: ${["Fighter", "Tank", "Healer"][selectedRole]}`
                    }
                  </div>
                </div>
              ) : isRevealStage ? (
                /* Battle Results */
                <div className="animate-fadeIn">
                  <div className="bg-gray-800 text-white rounded-lg px-4 py-3 text-center mb-4">
                    <h3 className="text-xl font-bold">Round {roundNumber} Results</h3>
                  </div>

                  {/* Health Changes Summary */}
                  <div className="grid grid-cols-2 gap-4 mb-4">
                    {/* Enemy Health Change */}
                    <div className="bg-red-50 border-2 border-red-400 rounded-lg p-4">
                      <div className="text-xs font-semibold text-gray-600 mb-2 uppercase tracking-wide">Enemy Health</div>
                      <div className="flex items-center justify-center gap-2 mb-2">
                        <span className="text-2xl font-bold text-gray-700">{previousEnemyHealth}</span>
                        <span className="text-xl text-gray-400">→</span>
                        <span className="text-2xl font-bold text-red-600">{enemyHealth}</span>
                      </div>
                      {damageToEnemy > 0 && (
                        <div className="text-sm font-medium text-red-600">
                          -{damageToEnemy} damage dealt
                        </div>
                      )}
                    </div>

                    {/* Team Health Change */}
                    <div className="bg-green-50 border-2 border-green-400 rounded-lg p-4">
                      <div className="text-xs font-semibold text-gray-600 mb-2 uppercase tracking-wide">Team Health</div>
                      <div className="flex items-center justify-center gap-2 mb-2">
                        <span className="text-2xl font-bold text-gray-700">{previousTeamHealth}</span>
                        <span className="text-xl text-gray-400">→</span>
                        <span className={`text-2xl font-bold ${teamHealth > previousTeamHealth ? 'text-green-600' : teamHealth < previousTeamHealth ? 'text-orange-600' : 'text-gray-700'}`}>
                          {teamHealth}
                        </span>
                      </div>
                      <div className="text-sm font-medium space-y-1">
                        {damageToTeam > 0 && (
                          <div className="text-orange-600">-{damageToTeam} damage taken</div>
                        )}
                        {healAmount > 0 && (
                          <div className="text-green-600">+{healAmount} healing received</div>
                        )}
                        {damageToTeam === 0 && healAmount === 0 && (
                          <div className="text-gray-500">No change</div>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Action Summary */}
                  <div className="bg-blue-50 border-2 border-blue-300 rounded-lg p-3 mb-4">
                    <div className="text-xs font-semibold text-gray-600 mb-2 uppercase tracking-wide">What Happened This Round</div>

                    {/* Team Actions */}
                    <div className="mb-3">
                      <div className="text-xs text-gray-500 mb-1">Team Actions:</div>
                      <div className="flex justify-center gap-4">
                        {actions.map((action, idx) => (
                          <div key={idx} className="flex flex-col items-center">
                            <div className="text-3xl mb-1">{actionIcons[action]}</div>
                            <div className="text-xs text-gray-600">{players[idx]?.id === player.id ? "YOU" : `P${idx + 1}`}</div>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Enemy Action */}
                    <div className="border-t border-blue-200 pt-2">
                      <div className="text-xs text-gray-500 mb-1">Enemy Action:</div>
                      <div className="flex justify-center items-center gap-2">
                        <div className="text-3xl">{enemyIntent === "WILL_ATTACK" ? "⚔️" : "😴"}</div>
                        <div className="text-sm font-medium text-gray-700">
                          {enemyIntent === "WILL_ATTACK" ? "Enemy attacked!" : "Enemy did nothing"}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="text-center text-gray-500 text-sm">
                    {countdown !== null && countdown > 0 ? (
                      <div className="text-lg font-bold text-blue-600">
                        Next round in {countdown}...
                      </div>
                    ) : countdown === 0 ? (
                      <div className="text-lg font-bold text-green-600">
                        Starting now!
                      </div>
                    ) : (
                      "Next round starting soon..."
                    )}
                  </div>
                </div>
              ) : (
                /* Action Menu */
                <div>
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
                          : isRoleCommitted
                            ? "bg-green-600 hover:bg-green-700 shadow-lg"
                            : "bg-blue-600 hover:bg-blue-700 shadow-lg"
                      }`}
                    >
                      {isRoleCommitted
                        ? `▶ READY (${roundsRemaining} rounds left)`
                        : selectedRole === null
                          ? "Select a role"
                          : "✓ CONFIRM ROLE (3 rounds)"}
                    </button>
                  </div>
                </div>
              )}
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

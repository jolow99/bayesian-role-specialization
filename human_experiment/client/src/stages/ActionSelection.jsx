import React, { useState, useEffect, useMemo, useCallback } from "react";
import { usePlayer, useGame, useRound, usePlayers, useStage } from "@empirica/core/player/classic/react";
import { BattleField } from "../components/BattleField";
import { ActionMenu } from "../components/ActionMenu";
import { ResultsPanel } from "../components/ResultsPanel";
import { ActionHistory } from "../components/ActionHistory";

const ROLES = { FIGHTER: 0, TANK: 1, HEALER: 2 };
const EMPTY_ARRAY = []; // Stable reference to prevent unnecessary re-renders

let renderCount = 0;

function ActionSelection() {
  renderCount++;
  console.log(`[RENDER #${renderCount}] ActionSelection component rendering`);

  const player = usePlayer();
  const players = usePlayers();
  const game = useGame();
  const round = useRound();
  const stage = useStage();

  const [selectedRole, setSelectedRole] = useState(null);
  const [showDamageAnimation, setShowDamageAnimation] = useState(false);
  const [countdown, setCountdown] = useState(null);

  // Cache the last valid data to show during transitions
  const lastValidDataRef = React.useRef(null);

  // Get current data from Empirica
  // Note: submitted is always read fresh (not cached) since it's stage-specific
  const treatment = game.get("treatment");
  const submitted = player.stage.get("submit");
  const rawEnemyHealth = game.get("enemyHealth");
  const rawTeamHealth = game.get("teamHealth");
  const rawEnemyIntent = round.get("enemyIntent");
  const rawRoundNumber = round.get("roundNumber");
  const rawMaxRounds = treatment?.maxRounds;
  const rawMaxHealth = treatment?.maxTeamHealth;
  const rawMaxEnemyHealth = treatment?.maxEnemyHealth;
  const rawCurrentStage = stage.get("name");
  const rawActions = round.get("actions");

  // Check if we have valid data
  const hasValidData = rawRoundNumber !== undefined && rawRoundNumber !== null;

  // Use cached data during transitions, or update cache with new valid data
  let enemyHealth, teamHealth, enemyIntent, roundNumber, maxRounds, maxHealth, maxEnemyHealth, currentStage, isRevealStage, actions;

  if (hasValidData) {
    // Valid data - use it and update cache
    enemyHealth = rawEnemyHealth;
    teamHealth = rawTeamHealth;
    enemyIntent = rawEnemyIntent;
    roundNumber = rawRoundNumber;
    maxRounds = rawMaxRounds;
    maxHealth = rawMaxHealth;
    maxEnemyHealth = rawMaxEnemyHealth;
    currentStage = rawCurrentStage;
    isRevealStage = currentStage === "Reveal";
    actions = rawActions || EMPTY_ARRAY;

    // Update cache (note: submitted is NOT cached)
    lastValidDataRef.current = {
      enemyHealth, teamHealth, enemyIntent, roundNumber,
      maxRounds, maxHealth, maxEnemyHealth, currentStage, isRevealStage, actions
    };
  } else if (lastValidDataRef.current) {
    // Invalid data but we have cache - use cached values
    console.log('[Client] Using cached data during transition');
    ({ enemyHealth, teamHealth, enemyIntent, roundNumber,
       maxRounds, maxHealth, maxEnemyHealth, currentStage, isRevealStage, actions } = lastValidDataRef.current);
  } else {
    // No valid data and no cache - first render, just return null
    console.log('[Client] No valid data available yet');
    return null;
  }

  // Get round-specific data (also cache these during transitions)
  const rawDamageToEnemy = round.get("damageToEnemy");
  const rawDamageToTeam = round.get("damageToTeam");
  const rawHealAmount = round.get("healAmount");
  const rawPreviousEnemyHealth = round.get("previousEnemyHealth");
  const rawPreviousTeamHealth = round.get("previousTeamHealth");

  let damageToEnemy, damageToTeam, healAmount, previousEnemyHealth, previousTeamHealth;

  if (hasValidData) {
    damageToEnemy = rawDamageToEnemy || 0;
    damageToTeam = rawDamageToTeam || 0;
    healAmount = rawHealAmount || 0;
    previousEnemyHealth = rawPreviousEnemyHealth || enemyHealth;
    previousTeamHealth = rawPreviousTeamHealth || teamHealth;

    // Add to cache
    lastValidDataRef.current = {
      ...lastValidDataRef.current,
      damageToEnemy, damageToTeam, healAmount, previousEnemyHealth, previousTeamHealth
    };
  } else if (lastValidDataRef.current) {
    ({ damageToEnemy, damageToTeam, healAmount, previousEnemyHealth, previousTeamHealth } = lastValidDataRef.current);
  } else {
    // Defaults for first render
    damageToEnemy = 0;
    damageToTeam = 0;
    healAmount = 0;
    previousEnemyHealth = enemyHealth;
    previousTeamHealth = teamHealth;
  }

  // Role commitment state
  const currentRole = player.get("currentRole");
  const roleEndRound = player.get("roleEndRound");
  const isRoleCommitted = currentRole !== null;
  const roundsRemaining = isRoleCommitted ? (roleEndRound - roundNumber + 1) : 0;

  // Debug logging
  console.log(`[Client RENDER] hasValidData: ${hasValidData}`);
  console.log(`[Client RENDER] Round ${roundNumber}, Stage: ${currentStage}, isRevealStage: ${isRevealStage}, submitted: ${submitted}`);
  console.log(`[Client RENDER] Enemy HP: ${enemyHealth}, Team HP: ${teamHealth}`);
  console.log(`[Client RENDER] isRoleCommitted: ${isRoleCommitted}, currentRole: ${currentRole}, roleEndRound: ${roleEndRound}`);
  console.log(`[Client RENDER] UI: waiting=${(submitted || isRoleCommitted) && !isRevealStage}, actionMenu=${!submitted && !isRevealStage && !isRoleCommitted}, reveal=${isRevealStage}`);

  // Debug: Track round changes and component lifecycle
  const prevRoundRef = React.useRef(null);
  const mountTimeRef = React.useRef(Date.now());

  // // Don't render during round transition if we don't have valid round data
  // if (roundNumber === undefined || roundNumber === null) {
  //   return null;
  // }

  useEffect(() => {
    console.log(`[COMPONENT MOUNTED] at ${Date.now()}, mountTime: ${mountTimeRef.current}`);
    return () => {
      console.log(`[COMPONENT UNMOUNTING]`);
    };
  }, []); // Empty deps = runs on mount/unmount only

  useEffect(() => {
    if (hasValidData && prevRoundRef.current !== roundNumber) {
      console.log(`[CLIENT ROUND CHANGE] ${prevRoundRef.current} → ${roundNumber}`);
      prevRoundRef.current = roundNumber;
    }
  }, [roundNumber, hasValidData]);

  // Auto-submit is now handled server-side in callbacks.js onStageStart
  // Server auto-submits players with committed roles when Action Selection stage starts

  // Trigger damage animation during reveal
  useEffect(() => {
    // Don't run effects during data transitions
    if (!hasValidData) return;

    if (isRevealStage) {
      setShowDamageAnimation(true);
      const timer = setTimeout(() => setShowDamageAnimation(false), 12000); // Extended to 12 seconds for 15-second reveal
      return () => clearTimeout(timer);
    }
  }, [isRevealStage, roundNumber, hasValidData]);

  // Auto-submit after reveal stage duration (15 seconds)
  useEffect(() => {
    // Don't run effects during data transitions
    if (!hasValidData) return;

    if (isRevealStage && !submitted) {
      console.log(`[Reveal] Auto-submitting after 15 seconds`);
      const timer = setTimeout(() => {
        console.log(`[Reveal] NOW submitting`);
        player.stage.set("submit", true);
      }, 15000); // Submit after 15 seconds
      return () => clearTimeout(timer);
    }
  }, [isRevealStage, submitted, player, hasValidData]);

  // Countdown timer for the last 5 seconds of reveal
  useEffect(() => {
    // Don't run effects during data transitions
    if (!hasValidData) return;

    if (isRevealStage) {
      // Start countdown at 10 seconds (showing countdown for last 5 seconds)
      const countdownStart = setTimeout(() => {
        setCountdown(5);
      }, 10000);

      return () => clearTimeout(countdownStart);
    } else {
      setCountdown(null);
    }
  }, [isRevealStage, roundNumber, hasValidData]);

  // Update countdown every second
  useEffect(() => {
    // Don't run effects during data transitions
    if (!hasValidData) return;

    if (countdown !== null && countdown > 0) {
      const timer = setTimeout(() => {
        setCountdown(countdown - 1);
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [countdown, hasValidData]);

  const handleRoleSelect = useCallback((role) => {
    if (!submitted && !isRoleCommitted) {
      setSelectedRole(role);
    }
  }, [submitted, isRoleCommitted]);

  const handleSubmit = useCallback(() => {
    if (!submitted) {
      if (!isRoleCommitted && selectedRole !== null) {
        // New role selection
        player.round.set("selectedRole", selectedRole);
      }
      // Always submit (whether role is committed or newly selected)
      player.stage.set("submit", true);
    }
  }, [submitted, isRoleCommitted, selectedRole, player]);

  // Get virtual bots from game state
  const virtualBots = game.get("virtualBots") || EMPTY_ARRAY;
  const totalPlayers = treatment?.totalPlayers;

  // Build unified player array (real + virtual) - fully cached for stability
  // Cache both the array AND the individual player objects to prevent any re-renders
  const allPlayersRef = React.useRef(null);
  const playerEntriesRef = React.useRef({});

  // Build current player array using Array constructor for stability
  const currentPlayers = new Array(totalPlayers).fill(null);

  // Add real players - reuse cached entry if player object is the same
  players.forEach(p => {
    const playerId = p.get("playerId");
    const cacheKey = `real-${playerId}`;
    const cached = playerEntriesRef.current[cacheKey];

    // CRITICAL: Only reuse if the actual player object is the same
    if (cached && cached.player === p) {
      currentPlayers[playerId] = cached;  // Reuse exact same object reference
    } else {
      const newEntry = { type: "real", player: p, playerId };
      currentPlayers[playerId] = newEntry;
      playerEntriesRef.current[cacheKey] = newEntry;
    }
  });

  // Add virtual bots - deep compare stats, reuse if match
  virtualBots.forEach(bot => {
    const cacheKey = `virtual-${bot.playerId}`;
    const cached = playerEntriesRef.current[cacheKey];

    // Deep compare bot object (check if stats and role state are the same)
    const botMatches = cached && cached.bot &&
      cached.bot.playerId === bot.playerId &&
      cached.bot.currentRole === bot.currentRole &&
      cached.bot.roleEndRound === bot.roleEndRound &&  // FIX: was missing, causing cache misses
      cached.bot.stats.STR === bot.stats.STR &&
      cached.bot.stats.DEF === bot.stats.DEF &&
      cached.bot.stats.SUP === bot.stats.SUP;

    if (botMatches) {
      currentPlayers[bot.playerId] = cached;  // Reuse exact same object reference
    } else {
      const newEntry = { type: "virtual", bot, playerId: bot.playerId };
      currentPlayers[bot.playerId] = newEntry;
      playerEntriesRef.current[cacheKey] = newEntry;
    }
  });

  // Only update ref if structure actually changed (check entry identity)
  const playersChanged = !allPlayersRef.current ||
    allPlayersRef.current.length !== currentPlayers.length ||
    allPlayersRef.current.some((entry, idx) => entry !== currentPlayers[idx]);

  if (playersChanged) {
    console.log(`[DEBUG] Player array structure changed, updating ref`);
    allPlayersRef.current = currentPlayers;
  }

  const allPlayers = allPlayersRef.current;

  // Determine which UI to show based on state
  let currentUI;
  if (isRevealStage) {
    currentUI = 'reveal';
  } else if (submitted || isRoleCommitted) {
    currentUI = 'waiting';
  } else {
    currentUI = 'actionMenu';
  }

  return (
    <div className="fixed inset-0 bg-gradient-to-b from-blue-400 to-blue-600 flex items-center justify-center p-4">
      <div className="w-full max-w-6xl" style={{ height: 'calc(100vh - 32px)' }}>
        {/* Battle Screen */}
        <div className="bg-white rounded-lg shadow-2xl border-4 border-gray-800 h-full flex flex-col">
          {/* Round Header - Fixed height */}
          <div className="bg-gray-800 text-white text-center flex-shrink-0" style={{ height: '60px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <h1 className="text-xl font-bold">Round {roundNumber}/{maxRounds}</h1>
            {isRoleCommitted && (
              <p className="text-xs text-yellow-300">
                Role: {["Fighter", "Tank", "Healer"][currentRole]} ({roundsRemaining} rounds left)
              </p>
            )}
          </div>

          {/* Battle Field - Fixed height */}
          <div className="flex-shrink-0" style={{ height: '320px' }}>
            <BattleField
              enemyHealth={enemyHealth}
              maxEnemyHealth={maxEnemyHealth}
              teamHealth={teamHealth}
              maxHealth={maxHealth}
              enemyIntent={enemyIntent}
              isRevealStage={isRevealStage}
              showDamageAnimation={showDamageAnimation}
              damageToEnemy={damageToEnemy}
              damageToTeam={damageToTeam}
              healAmount={healAmount}
              actions={actions}
              allPlayers={allPlayers}
              currentPlayerId={player.id}
            />
          </div>

          {/* Battle Results (during Reveal) or Action Menu (during Action Selection) - Fixed height */}
          <div className="bg-white border-t-4 border-gray-700 flex-shrink-0 relative" style={{ height: '240px' }}>
            <div className="p-4 h-full overflow-auto">
              {/* Conditional rendering instead of opacity toggling */}
              {currentUI === 'waiting' && (
                <div className="text-center py-6">
                  <div className="text-4xl mb-3">⏳</div>
                  <div className="text-lg font-bold text-gray-700 mb-2">Waiting for other players...</div>
                  <div className="text-gray-500 text-sm">
                    {isRoleCommitted
                      ? `Your role: ${["Fighter", "Tank", "Healer"][currentRole]} (${roundsRemaining} rounds remaining)`
                      : selectedRole !== null ? `Your role: ${["Fighter", "Tank", "Healer"][selectedRole]}` : ''
                    }
                  </div>
                </div>
              )}

              {currentUI === 'reveal' && (
                <ResultsPanel
                  roundNumber={roundNumber}
                  enemyHealth={enemyHealth}
                  previousEnemyHealth={previousEnemyHealth}
                  damageToEnemy={damageToEnemy}
                  teamHealth={teamHealth}
                  previousTeamHealth={previousTeamHealth}
                  damageToTeam={damageToTeam}
                  healAmount={healAmount}
                  actions={actions}
                  allPlayers={allPlayers}
                  currentPlayerId={player.id}
                  enemyIntent={enemyIntent}
                  countdown={countdown}
                />
              )}

              {currentUI === 'actionMenu' && (
                <ActionMenu
                  selectedRole={selectedRole}
                  onRoleSelect={handleRoleSelect}
                  onSubmit={handleSubmit}
                  isRoleCommitted={isRoleCommitted}
                  currentRole={currentRole}
                  roundsRemaining={roundsRemaining}
                  submitted={submitted}
                />
              )}
            </div>
          </div>

          {/* Battle History and Game Info - Takes remaining space */}
          <div className="bg-gray-50 border-t-2 border-gray-300 flex-1 min-h-0 overflow-hidden">
            <div className="grid grid-cols-2 gap-3 h-full p-3">
              {/* Battle History */}
              <div className="bg-white rounded-lg border-2 border-gray-400 overflow-hidden flex flex-col">
                <div className="p-3 border-b border-gray-300 flex-shrink-0">
                  <h3 className="text-sm font-bold text-gray-800 flex items-center gap-2">
                    📜 Battle History
                  </h3>
                </div>
                <div className="flex-1 overflow-auto p-3 pt-2">
                  <ActionHistory />
                </div>
              </div>

              {/* Game Mechanics Info */}
              <div className="bg-white rounded-lg border-2 border-blue-400 overflow-hidden flex flex-col">
                <div className="p-3 border-b border-blue-300 flex-shrink-0">
                  <h3 className="text-sm font-bold text-gray-800 flex items-center gap-2">
                    ℹ️ How Stats Influence Actions
                  </h3>
                </div>
                <div className="flex-1 overflow-auto p-3 pt-2">
                  <div className="space-y-2 text-xs">
                    <div>
                      <span className="font-semibold">Attacks:</span> STR stats of all attack actions add together
                    </div>
                    <div>
                      <span className="font-semibold">Defending:</span> Max DEF stat of all defend actions
                    </div>
                    <div>
                      <span className="font-semibold">Healing:</span> SUP stats of all heal actions add together (up to max HP)
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Wrap with React.memo to prevent parent re-renders from cascading down
// Empirica hooks use reactive subscriptions that bypass memo, so this only prevents prop-based re-renders
export default React.memo(ActionSelection, () => {
  // ActionSelection has no props, so always prevent re-render from parent
  // Hooks will still trigger re-renders automatically
  return true;
});

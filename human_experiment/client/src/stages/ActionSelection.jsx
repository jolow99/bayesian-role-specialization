import React, { useState, useEffect, useMemo, useCallback } from "react";
import { usePlayer, useGame, useRound, usePlayers, useStage } from "@empirica/core/player/classic/react";
import { BattleField } from "../components/BattleField";
import { ActionMenu } from "../components/ActionMenu";
import { ResultsPanel } from "../components/ResultsPanel";
import { ActionHistory } from "../components/ActionHistory";
import { GameEndScreen } from "../components/GameEndScreen";
import { ROLES, ROLE_LABELS } from "../constants";

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

  const containerRef = React.useRef(null);

  const [selectedRole, setSelectedRole] = useState(null);
  const [showDamageAnimation, setShowDamageAnimation] = useState(false);
  const [localSubmitted, setLocalSubmitted] = useState(false); // Local state to immediately show waiting screen
  const [currentTurnView, setCurrentTurnView] = useState(0); // 0 = role selection/waiting, 1 = turn 1 results, 2 = turn 2 results
  const [lastViewedStage, setLastViewedStage] = useState(0); // Track which stage's turns we've viewed
  const [allowRoundEnd, setAllowRoundEnd] = useState(false); // Control when to show round end screen

  // Track if player clicked continue after early finish - stored in Empirica state to persist across remounts
  const acknowledgedEarlyFinish = player.round.get("acknowledgedEarlyFinish") || false;
  const setAcknowledgedEarlyFinish = useCallback((value) => {
    player.round.set("acknowledgedEarlyFinish", value);
  }, [player]);

  // Cache the current stage state HTML continuously (after every render)
  // This ensures the cache is ALWAYS up-to-date before any unmount happens
  useEffect(() => {
    if (containerRef.current) {
      try {
        const html = containerRef.current.outerHTML; // Use outerHTML to include the container itself
        sessionStorage.setItem('stageStateCache', JSON.stringify({ html }));
        // Don't log on every render to avoid spam
      } catch (e) {
        console.error('[ActionSelection] Failed to cache state:', e);
      }
    }
  }); // No dependency array - runs after every render

  // Clear stage cache on mount (new stage state has loaded)
  useEffect(() => {
    console.log('[ActionSelection] Mounted - will use cached state during transition');
    // Clear cache after a delay to allow transition to complete
    const timer = setTimeout(() => {
      sessionStorage.removeItem('stageStateCache');
      console.log('[ActionSelection] Cleared stage cache after transition complete');
    }, 200);
    return () => clearTimeout(timer);
  }, []);

  // Get current data from Empirica
  const treatment = game.get("treatment");
  const roundNumber = round.get("roundNumber");
  const roundConfig = roundNumber ? game.get(`round${roundNumber}Config`) : null;

  // Initialize randomized role order once per player for the entire game
  useEffect(() => {
    if (!player.get("roleOrder")) {
      // Generate random permutation of [0, 1, 2]
      const roles = [ROLES.FIGHTER, ROLES.TANK, ROLES.MEDIC];
      const shuffled = [...roles].sort(() => Math.random() - 0.5);
      player.set("roleOrder", shuffled);
      console.log("[ActionSelection] Initialized random role order:", shuffled);
    }
  }, [player]);

  const roleOrder = player.get("roleOrder") || [ROLES.FIGHTER, ROLES.TANK, ROLES.MEDIC];
  const submitted = player.stage.get("submit") || localSubmitted; // Use local state during delay period

  const stageNumber = stage.get("stageNumber");
  const maxRounds = treatment?.maxRounds;
  const maxStagesPerRound = treatment?.maxStagesPerRound;
  const currentStage = stage.get("name");
  const stageType = stage.get("stageType");

  // Check if we have valid data
  if (roundNumber === undefined || roundNumber === null) {
    return null;
  }

  // Derive isBotRound from shuffledRoundOrder (same logic as server)
  // This is more reliable than round.get("isBotRound") due to Empirica sync timing
  const shuffledRoundOrder = game.get("shuffledRoundOrder") || [];
  const roundSlot = shuffledRoundOrder[roundNumber - 1];
  const isBotRound = roundSlot?.type === "bot";

  // For bot rounds, use player-specific config; for human rounds, use shared round config
  const playerRoundConfig = player.round.get("playerRoundConfig");
  const effectiveConfig = isBotRound ? playerRoundConfig : roundConfig;
  const maxHealth = effectiveConfig?.maxTeamHealth;
  const maxEnemyHealth = effectiveConfig?.maxEnemyHealth;

  console.log(`[Config Debug] isBotRound: ${isBotRound}, playerRoundConfig:`, playerRoundConfig, `roundConfig:`, roundConfig, `effectiveConfig:`, effectiveConfig);

  // Get turns data (computed server-side after role selection)
  // In bot rounds, read from player-specific state; in human rounds, from shared round state
  const currentRoundStageNumber = round.get("stageNumber") || 0;
  const stageToView = Math.max(lastViewedStage + 1, currentRoundStageNumber);
  const turns = isBotRound
    ? (player.round.get(`stage${stageToView}Turns`) || EMPTY_ARRAY)
    : (round.get(`stage${stageToView}Turns`) || EMPTY_ARRAY);
  const hasTurns = turns.length > 0;

  console.log(`[Turns Data] isBotRound: ${isBotRound}, currentRoundStageNumber: ${currentRoundStageNumber}, lastViewedStage: ${lastViewedStage}, stageToView: ${stageToView}, hasTurns: ${hasTurns}`);

  // Read health - if we're viewing a specific turn, show health after that turn
  // In bot rounds, use per-player health; in human rounds, use shared round health
  let enemyHealth, teamHealth;
  if (hasTurns && currentTurnView > 0 && currentTurnView <= turns.length) {
    const turn = turns[currentTurnView - 1];
    enemyHealth = turn.newEnemyHealth;
    teamHealth = turn.newTeamHealth;
  } else {
    // Default to round/player health or config defaults
    if (isBotRound) {
      enemyHealth = player.round.get("enemyHealth") ?? effectiveConfig?.maxEnemyHealth;
      teamHealth = player.round.get("teamHealth") ?? effectiveConfig?.maxTeamHealth;
    } else {
      enemyHealth = round.get("enemyHealth") ?? roundConfig?.maxEnemyHealth;
      teamHealth = round.get("teamHealth") ?? roundConfig?.maxTeamHealth;
    }
  }

  // Determine which turn's data to show based on currentTurnView
  let enemyIntent, actions, damageToEnemy, damageToTeam, damageBlocked, healAmount, previousEnemyHealth, previousTeamHealth;

  if (hasTurns && currentTurnView > 0 && currentTurnView <= turns.length) {
    // Show data for the current turn view
    const turn = turns[currentTurnView - 1];
    enemyIntent = turn.enemyIntent;
    actions = turn.actions || EMPTY_ARRAY;
    damageToEnemy = turn.damageToEnemy || 0;
    damageToTeam = turn.damageToTeam || 0;
    damageBlocked = turn.damageBlocked || 0;
    healAmount = turn.healAmount || 0;
    previousEnemyHealth = turn.previousEnemyHealth || enemyHealth;
    previousTeamHealth = turn.previousTeamHealth || teamHealth;
  } else {
    // Role selection or waiting - no turn data yet
    enemyIntent = null;
    actions = EMPTY_ARRAY;
    damageToEnemy = 0;
    damageToTeam = 0;
    damageBlocked = 0;
    healAmount = 0;
    previousEnemyHealth = enemyHealth;
    previousTeamHealth = teamHealth;
  }

  // Determine stage types
  const isRoundEndStage = stageType === "roundEnd";
  const isGameEndStage = stageType === "gameEnd";
  const isTurnStage = hasTurns && currentTurnView > 0;

  // Check if round has ended - in bot rounds, use per-player outcome
  const roundOutcome = isBotRound ? player.round.get("outcome") : round.get("outcome");
  const shouldShowRoundEnd = (isRoundEndStage || (roundOutcome && !isGameEndStage)) && allowRoundEnd;

  // Check if game has ended
  // Use player-specific totalPoints (each player tracks their own cumulative score)
  const totalPoints = player.get("totalPoints") || 0;
  const shouldShowGameEnd = isGameEndStage;

  // Debug logging
  console.log(`[Client RENDER] Round ${roundNumber}, Stage: ${currentStage}, Type: ${stageType}`);
  console.log(`[Client RENDER] Enemy HP: ${enemyHealth}, Team HP: ${teamHealth}`);
  console.log(`[Client RENDER] submitted: ${submitted}, hasTurns: ${hasTurns}, currentTurnView: ${currentTurnView}, turns.length: ${turns.length}`);

  // Reset last viewed stage and round end flag when moving to a new round
  // Note: acknowledgedEarlyFinish is stored in player.round state, so it auto-resets with new rounds
  useEffect(() => {
    setLastViewedStage(0);
    setAllowRoundEnd(false);
  }, [roundNumber]);

  // Reset localSubmitted when stage changes (but preserve acknowledgedEarlyFinish)
  const stageId = stage?.id;
  useEffect(() => {
    console.log(`[Stage Change] Stage ID changed to ${stageId}, resetting localSubmitted`);
    setLocalSubmitted(false);
  }, [stageId]);

  // Auto-submit for players who have finished their bot round early
  // This runs continuously while the player has acknowledged their early finish
  // Includes round end stage (so they don't have to click Continue again)
  // but stops at game end stage (they need to see final results)
  useEffect(() => {
    if (!isBotRound || !roundOutcome || !acknowledgedEarlyFinish || isGameEndStage) {
      return;
    }

    // Set up an interval to keep checking and submitting
    // This handles cases where the stage changes but the effect doesn't re-run immediately
    const submitIfNeeded = () => {
      const currentSubmitStatus = player.stage.get("submit");
      if (!currentSubmitStatus) {
        console.log(`[Auto-Submit] Player finished bot round early with ${roundOutcome}, auto-submitting for stage ${stageId} (isRoundEndStage: ${isRoundEndStage})`);
        player.stage.set("submit", true);
      }
    };

    // Initial submit attempt after a small delay
    const initialTimer = setTimeout(submitIfNeeded, 100);

    // Keep checking in case the initial attempt didn't work (e.g., stage wasn't ready)
    const intervalTimer = setInterval(submitIfNeeded, 500);

    return () => {
      clearTimeout(initialTimer);
      clearInterval(intervalTimer);
    };
  }, [isBotRound, roundOutcome, acknowledgedEarlyFinish, isRoundEndStage, isGameEndStage, player, stageId]);

  // If we're entering a round end stage, allow round end screen
  // Also handle bot rounds where player may finish before round end stage is created
  useEffect(() => {
    // For bot rounds: if player has an outcome but we're not in round end stage yet,
    // still allow showing the round end screen after viewing turns
    const playerFinishedBotRound = isBotRound && roundOutcome && !isRoundEndStage;

    if ((isRoundEndStage || playerFinishedBotRound) && !allowRoundEnd) {
      if (!hasTurns) {
        // No turns to show, allow round end immediately
        console.log(`[Round End] Entered round end stage directly (no turns), allowing round end screen`);
        setAllowRoundEnd(true);
      } else if (currentTurnView === turns.length && turns.length > 0) {
        // We've already viewed all turns, allow round end immediately
        console.log(`[Round End] Entered round end stage after viewing all turns, allowing round end screen`);
        setAllowRoundEnd(true);
      } else if (currentTurnView === 0 && turns.length > 0) {
        // We haven't started viewing turns yet, start viewing them
        console.log(`[Round End] Entered round end stage with unviewed turns, starting turn display`);
        setCurrentTurnView(1);
      }
    }
  }, [isRoundEndStage, hasTurns, allowRoundEnd, currentTurnView, turns.length, isBotRound, roundOutcome]);

  // Auto-start showing turn 1 when turns data arrives
  useEffect(() => {
    if (hasTurns && currentTurnView === 0 && !isRoundEndStage && !isGameEndStage) {
      console.log(`[Turn Auto-Advance] Turns data arrived, starting turn 1 display`);
      setCurrentTurnView(1);
    }
  }, [hasTurns, currentTurnView, isRoundEndStage, isGameEndStage]);



  // Trigger damage animation during turn stages
  useEffect(() => {
    if (isTurnStage) {
      setShowDamageAnimation(true);
      const timer = setTimeout(() => setShowDamageAnimation(false), 7000); // Show animation for 7 seconds
      return () => clearTimeout(timer);
    }
  }, [isTurnStage, currentTurnView]);


  // Handle manual advancement through turns
  const handleNextTurn = useCallback(() => {
    if (currentTurnView < turns.length) {
      // Advance to next turn
      console.log(`[Turn Advance] User clicked next, advancing from turn ${currentTurnView} to ${currentTurnView + 1}`);
      setCurrentTurnView(currentTurnView + 1);
    } else if (currentTurnView === turns.length && turns.length > 0) {
      // All turns viewed - either show round end or go to next stage
      if (roundOutcome) {
        console.log(`[Turn Advance] All turns viewed, round ended with outcome: ${roundOutcome}, showing round end`);
        setAllowRoundEnd(true);
      } else {
        console.log(`[Turn Advance] All turns viewed, marking stage ${stageToView} as viewed, resetting to role selection`);
        setLastViewedStage(stageToView);
        setCurrentTurnView(0);
        setLocalSubmitted(false);
      }
    }
  }, [currentTurnView, turns.length, roundOutcome, stageToView]);

  const handleRoleSelect = useCallback((role) => {
    if (!submitted && !hasTurns) {
      setSelectedRole(role);
    }
  }, [submitted, hasTurns]);

  const handleSubmit = useCallback(() => {
    if (!submitted && selectedRole !== null) {
      // Store the selected role on stage (not round)
      player.stage.set("selectedRole", selectedRole);
      player.stage.set("roleSubmittedAt", Date.now()); // Track when role was submitted

      // Set local state to immediately show waiting screen
      setLocalSubmitted(true);

      // Submit immediately - realism comes from waiting for other real humans
      // who are also playing with bots in their own sessions
      player.stage.set("submit", true);
    }
  }, [submitted, selectedRole, player]);

  // Get virtual bots from player-specific round state (each player has their own bot positions)
  // Falls back to round-level virtualBots for backward compatibility
  const virtualBots = player.round.get("virtualBots") || round.get("virtualBots") || EMPTY_ARRAY;
  const totalPlayers = treatment?.totalPlayers;
  const hasBots = virtualBots.length > 0;

  // Build unified player array (real + virtual)
  const allPlayers = new Array(totalPlayers).fill(null);

  // In bot rounds, current player is at their position, bots fill the rest
  // In human rounds, all real players are at their positions
  if (hasBots) {
    // Bot round: current player + virtual bots
    const currentPlayerId = player.get("gamePlayerId");
    allPlayers[currentPlayerId] = { type: "real", player, playerId: currentPlayerId };

    virtualBots.forEach(bot => {
      allPlayers[bot.playerId] = { type: "virtual", bot, playerId: bot.playerId };
    });
  } else {
    // Human round: add all real players using their permanent gamePlayerId
    players.forEach(p => {
      const playerId = p.get("gamePlayerId"); // Use permanent game-level ID
      if (playerId !== null && playerId !== undefined) {
        allPlayers[playerId] = { type: "real", player: p, playerId };
      }
    });
  }

  // Get current player's permanent game ID (0, 1, or 2)
  const currentPlayerGameId = player.get("gamePlayerId");

  // Track submission status for other players (excluding current player)
  // In bot rounds: show bots at their positions, but use real human submission status
  // In human rounds: show other real players
  const otherPlayersStatus = useMemo(() => {
    if (hasBots) {
      // Bot round: show virtual bots, but their "submitted" status comes from real humans
      // We map bot positions to the other human players' submission status
      const otherHumans = players.filter(p => p.id !== player.id);

      return virtualBots.map((bot, idx) => ({
        odId: `bot-${bot.playerId}`,
        playerId: bot.playerId,
        isBot: true,
        // Use corresponding human's submission status (or true if no more humans)
        submitted: otherHumans[idx]?.stage.get("submit") || otherHumans.length === 0
      })).sort((a, b) => (a.playerId ?? 0) - (b.playerId ?? 0));
    } else {
      // Human round: show other real players
      return players
        .filter(p => p.id !== player.id)
        .map(p => ({
          odId: p.id,
          playerId: p.get("gamePlayerId"), // Permanent game position (0, 1, or 2)
          isBot: false,
          submitted: p.stage.get("submit") || false
        }))
        .sort((a, b) => (a.playerId ?? 0) - (b.playerId ?? 0)); // Sort by playerId for consistent order
    }
  }, [players, player.id, hasBots, virtualBots]);

  // Determine which UI to show based on state
  // Note: roundEnd and gameEnd are now overlays, so we show the underlying UI beneath them
  let currentUI;
  if (isGameEndStage) {
    // Show last turn results beneath the game end overlay (if available)
    currentUI = currentTurnView > 0 ? 'turnResults' : 'waiting';
  } else if (isRoundEndStage) {
    // Show last turn results beneath the round end overlay (if available)
    currentUI = currentTurnView > 0 ? 'turnResults' : 'waiting';
  } else if (isTurnStage) {
    currentUI = 'turnResults';
  } else if (submitted) {
    currentUI = 'waiting';
  } else {
    currentUI = 'roleSelection';
  }

  return (
    <div ref={containerRef} className="fixed inset-0 bg-gradient-to-b from-blue-400 to-blue-600 flex items-center justify-center p-2">
      <div className="w-full h-full flex items-center justify-center" style={{ maxWidth: '1400px' }}>
        {/* Battle Screen */}
        <div className="bg-white rounded-lg shadow-2xl border-4 border-gray-800 w-full h-full flex overflow-hidden relative">
          {/* Left Column - Game Interface and Role Selection */}
          <div className="flex-1 flex flex-col min-w-0">
            {/* Round Header */}
            <div className="bg-gray-800 text-white text-center flex-shrink-0 rounded-tl-lg flex items-center justify-center" style={{ height: '40px' }}>
              <h1 className="text-lg font-bold">
                Round {roundNumber}/{maxRounds}
                {stageNumber > 0 && ` - Stage ${stageNumber}/${maxStagesPerRound}`}
                {totalPoints !== undefined && ` | Points: ${totalPoints}`}
              </h1>
            </div>

            {/* Battle Field */}
            <div className="flex-shrink-0" style={{ height: '35vh', minHeight: '250px', maxHeight: '400px' }}>
              <BattleField
                enemyHealth={enemyHealth}
                maxEnemyHealth={maxEnemyHealth}
                teamHealth={teamHealth}
                maxHealth={maxHealth}
                enemyIntent={enemyIntent}
                isRevealStage={isTurnStage}
                showDamageAnimation={showDamageAnimation}
                damageToEnemy={damageToEnemy}
                damageToTeam={damageToTeam}
                healAmount={healAmount}
                actions={actions}
                allPlayers={allPlayers}
                currentPlayerGameId={currentPlayerGameId}
                previousEnemyHealth={previousEnemyHealth}
                previousTeamHealth={previousTeamHealth}
                bossDamage={effectiveConfig?.bossDamage}
                enemyAttackProbability={effectiveConfig?.enemyAttackProbability}
              />
            </div>

            {/* Role Selection or Turn Results */}
            <div className="bg-white border-t-4 border-gray-700 flex-1 min-h-0 flex flex-col">
              <div className="flex-1 p-4 flex items-center justify-center">
                {/* Waiting for other players after submitting role */}
                {currentUI === 'waiting' && (
                  <div className="text-center">
                    <div className="text-4xl mb-3">⏳</div>
                    <div className="text-lg font-bold text-gray-700 mb-2">Waiting for other players...</div>
                    <div className="text-gray-500 text-sm mb-4">
                      {selectedRole !== null && `Your selected role: ${ROLE_LABELS[selectedRole]}`}
                    </div>
                    {/* Show other players' submission status */}
                    <div className="flex justify-center gap-4 mt-2">
                      {otherPlayersStatus.map((p) => (
                        <div key={p.odId} className="flex items-center gap-2">
                          <span className="text-sm text-gray-600">P{(p.playerId ?? 0) + 1}:</span>
                          {p.submitted ? (
                            <span className="text-green-600 font-semibold">✓ Ready</span>
                          ) : (
                            <span className="text-orange-500 font-semibold animate-pulse">⏳ Choosing...</span>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Turn Results - showing what happened in this turn */}
                {currentUI === 'turnResults' && (
                  <div className="w-full">
                    <ResultsPanel
                      stageNumber={stageToView}
                      turnNumber={currentTurnView}
                      actions={actions}
                      allPlayers={allPlayers}
                      currentPlayerGameId={currentPlayerGameId}
                      enemyIntent={enemyIntent}
                      onNextTurn={handleNextTurn}
                      nextButtonLabel={
                        currentTurnView < turns.length
                          ? "Next Turn"
                          : roundOutcome
                            ? "Continue"
                            : "Next Stage"
                      }
                      previousTeamHealth={previousTeamHealth}
                      newTeamHealth={teamHealth}
                      previousEnemyHealth={previousEnemyHealth}
                      newEnemyHealth={enemyHealth}
                      damageToTeam={damageToTeam}
                      damageToEnemy={damageToEnemy}
                      damageBlocked={damageBlocked}
                      healAmount={healAmount}
                    />
                  </div>
                )}

                {/* Role Selection Menu */}
                {currentUI === 'roleSelection' && (
                  <div className="w-full max-w-4xl">
                    <ActionMenu
                      selectedRole={selectedRole}
                      onRoleSelect={handleRoleSelect}
                      onSubmit={handleSubmit}
                      isRoleCommitted={false}
                      currentRole={null}
                      roundsRemaining={0}
                      submitted={submitted}
                      roleOrder={roleOrder}
                      otherPlayersStatus={otherPlayersStatus}
                    />
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right Column - Battle History (full height) */}
          <div className="bg-gray-50 border-l-4 border-gray-700 overflow-hidden flex flex-col" style={{ width: '22%', minWidth: '280px', maxWidth: '350px' }} data-tutorial-id="battle-history">
            <div className="bg-gray-800 text-white text-center flex-shrink-0 rounded-tr-lg flex items-center justify-center" style={{ height: '40px' }}>
              <h3 className="text-sm font-bold">
                📜 Battle History
              </h3>
            </div>
            <div className="flex-1 overflow-auto p-3 bg-white">
              <ActionHistory currentStageView={stageToView} currentTurnView={currentTurnView} />
            </div>
          </div>

          {/* Round End Overlay - covers left panel only, leaves history visible */}
          {shouldShowRoundEnd && (
            <div className="absolute top-0 bottom-0 left-0 bg-black bg-opacity-60 flex items-center justify-center z-50" style={{ right: 'calc(22% + 4px)', minWidth: 'calc(100% - 354px)' }}>
              <GameEndScreen
                outcome={roundOutcome}
                endMessage={round.get("roundEndMessage")}
                totalPoints={totalPoints}
                roundOutcomes={player.get("roundOutcomes") || []}
                isBotRoundEarlyFinish={isBotRound && roundOutcome && !isRoundEndStage}
                onEarlyFinishContinue={() => setAcknowledgedEarlyFinish(true)}
                otherPlayersStatus={otherPlayersStatus}
              />
            </div>
          )}

          {/* Game End Overlay - covers left panel only, leaves history visible */}
          {shouldShowGameEnd && (
            <div className="absolute top-0 bottom-0 left-0 bg-black bg-opacity-60 flex items-center justify-center z-50" style={{ right: 'calc(22% + 4px)', minWidth: 'calc(100% - 354px)' }}>
              <GameEndScreen
                outcome={game.get("finalOutcome")}
                endMessage={stage.get("endMessage")}
                totalPoints={totalPoints}
                roundOutcomes={player.get("roundOutcomes") || []}
                otherPlayersStatus={otherPlayersStatus}
              />
            </div>
          )}
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

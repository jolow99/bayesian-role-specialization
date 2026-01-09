import React, { useState, useEffect, useMemo, useCallback } from "react";
import { usePlayer, useGame, useRound, usePlayers, useStage } from "@empirica/core/player/classic/react";
import { BattleField } from "../components/BattleField";
import { ActionMenu } from "../components/ActionMenu";
import { ResultsPanel } from "../components/ResultsPanel";
import { ActionHistory } from "../components/ActionHistory";
import { GameEndScreen } from "../components/GameEndScreen";

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

  const containerRef = React.useRef(null);

  const [selectedRole, setSelectedRole] = useState(null);
  const [showDamageAnimation, setShowDamageAnimation] = useState(false);
  const [countdown, setCountdown] = useState(null);
  const [localSubmitted, setLocalSubmitted] = useState(false); // Local state to immediately show waiting screen
  const [currentTurnView, setCurrentTurnView] = useState(0); // 0 = role selection/waiting, 1 = turn 1 results, 2 = turn 2 results
  const [lastViewedStage, setLastViewedStage] = useState(0); // Track which stage's turns we've viewed
  const [allowRoundEnd, setAllowRoundEnd] = useState(false); // Control when to show round end screen

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
      const roles = [ROLES.FIGHTER, ROLES.TANK, ROLES.HEALER];
      const shuffled = [...roles].sort(() => Math.random() - 0.5);
      player.set("roleOrder", shuffled);
      console.log("[ActionSelection] Initialized random role order:", shuffled);
    }
  }, [player]);

  const roleOrder = player.get("roleOrder") || [ROLES.FIGHTER, ROLES.TANK, ROLES.HEALER];
  const submitted = player.stage.get("submit") || localSubmitted; // Use local state during delay period

  const stageNumber = stage.get("stageNumber");
  const maxRounds = treatment?.maxRounds;
  const maxStagesPerRound = treatment?.maxStagesPerRound;
  const maxHealth = roundConfig?.maxTeamHealth;
  const maxEnemyHealth = roundConfig?.maxEnemyHealth;
  const currentStage = stage.get("name");
  const stageType = stage.get("stageType");

  // Check if we have valid data
  if (roundNumber === undefined || roundNumber === null) {
    return null;
  }

  // Get turns data (computed server-side after role selection)
  // Check for turns from the most recent completed stage
  const currentRoundStageNumber = round.get("stageNumber") || 0;
  const stageToView = Math.max(lastViewedStage + 1, currentRoundStageNumber);
  const turns = round.get(`stage${stageToView}Turns`) || EMPTY_ARRAY;
  const hasTurns = turns.length > 0;

  console.log(`[Turns Data] currentRoundStageNumber: ${currentRoundStageNumber}, lastViewedStage: ${lastViewedStage}, stageToView: ${stageToView}, hasTurns: ${hasTurns}`);

  // Read health - if we're viewing a specific turn, show health after that turn
  let enemyHealth, teamHealth;
  if (hasTurns && currentTurnView > 0 && currentTurnView <= turns.length) {
    const turn = turns[currentTurnView - 1];
    enemyHealth = turn.newEnemyHealth;
    teamHealth = turn.newTeamHealth;
  } else {
    // Default to round health or config defaults
    enemyHealth = round.get("enemyHealth") ?? roundConfig?.maxEnemyHealth;
    teamHealth = round.get("teamHealth") ?? roundConfig?.maxTeamHealth;
  }

  // Determine which turn's data to show based on currentTurnView
  let enemyIntent, actions, damageToEnemy, damageToTeam, healAmount, previousEnemyHealth, previousTeamHealth, roles;

  if (hasTurns && currentTurnView > 0 && currentTurnView <= turns.length) {
    // Show data for the current turn view
    const turn = turns[currentTurnView - 1];
    enemyIntent = turn.enemyIntent;
    actions = turn.actions || EMPTY_ARRAY;
    roles = turn.roles || EMPTY_ARRAY;
    damageToEnemy = turn.damageToEnemy || 0;
    damageToTeam = turn.damageToTeam || 0;
    healAmount = turn.healAmount || 0;
    previousEnemyHealth = turn.previousEnemyHealth || enemyHealth;
    previousTeamHealth = turn.previousTeamHealth || teamHealth;
  } else {
    // Role selection or waiting - no turn data yet
    enemyIntent = null;
    actions = EMPTY_ARRAY;
    roles = EMPTY_ARRAY;
    damageToEnemy = 0;
    damageToTeam = 0;
    healAmount = 0;
    previousEnemyHealth = enemyHealth;
    previousTeamHealth = teamHealth;
  }

  // Determine stage types
  const isRoundEndStage = stageType === "roundEnd";
  const isGameEndStage = stageType === "gameEnd";
  const isTurnStage = hasTurns && currentTurnView > 0;

  // Check if round has ended
  const roundOutcome = round.get("outcome");
  const shouldShowRoundEnd = (isRoundEndStage || (roundOutcome && !isGameEndStage)) && allowRoundEnd;

  // Check if game has ended
  const totalPoints = game.get("totalPoints");
  const shouldShowGameEnd = isGameEndStage;

  // Debug logging
  console.log(`[Client RENDER] Round ${roundNumber}, Stage: ${currentStage}, Type: ${stageType}`);
  console.log(`[Client RENDER] Enemy HP: ${enemyHealth}, Team HP: ${teamHealth}`);
  console.log(`[Client RENDER] submitted: ${submitted}, hasTurns: ${hasTurns}, currentTurnView: ${currentTurnView}, turns.length: ${turns.length}`);

  // Reset last viewed stage and round end flag when moving to a new round
  useEffect(() => {
    setLastViewedStage(0);
    setAllowRoundEnd(false);
  }, [roundNumber]);

  // If we're entering a round end stage directly (no turns to show), allow round end immediately
  useEffect(() => {
    if (isRoundEndStage && !hasTurns && !allowRoundEnd) {
      console.log(`[Round End] Entered round end stage directly, allowing round end screen`);
      setAllowRoundEnd(true);
    }
  }, [isRoundEndStage, hasTurns, allowRoundEnd]);

  // Auto-start showing turn 1 when turns data arrives
  useEffect(() => {
    if (hasTurns && currentTurnView === 0 && !isRoundEndStage && !isGameEndStage) {
      console.log(`[Turn Auto-Advance] Turns data arrived, starting turn 1 display`);
      setCurrentTurnView(1);
    }
  }, [hasTurns, currentTurnView, isRoundEndStage, isGameEndStage]);

  // Auto-advance from turn 1 to turn 2 after delay
  useEffect(() => {
    if (currentTurnView === 1 && turns.length >= 2) {
      console.log(`[Turn Auto-Advance] Scheduling advance to turn 2 in 8 seconds`);
      const timer = setTimeout(() => {
        console.log(`[Turn Auto-Advance] Advancing to turn 2`);
        setCurrentTurnView(2);
      }, 8000); // 8 seconds to view turn 1 results
      return () => clearTimeout(timer);
    }
  }, [currentTurnView, turns.length]);

  // After viewing all turns, mark stage as viewed and reset to role selection OR show round end
  useEffect(() => {
    if (currentTurnView === turns.length && turns.length > 0) {
      const roundOutcomeExists = round.get("outcome");

      if (roundOutcomeExists) {
        // Round has ended (win/loss/timeout), transition to round end screen after delay
        console.log(`[Turn Auto-Advance] All turns viewed, round ended with outcome: ${roundOutcomeExists}, showing round end in 8 seconds`);
        const timer = setTimeout(() => {
          console.log(`[Turn Auto-Advance] Enabling round end screen`);
          setAllowRoundEnd(true);
        }, 8000); // 8 seconds to view final turn before showing victory/defeat
        return () => clearTimeout(timer);
      } else {
        // Round continues, transition to next stage's role selection
        console.log(`[Turn Auto-Advance] All turns viewed, scheduling transition to next stage in 8 seconds`);
        const timer = setTimeout(() => {
          console.log(`[Turn Auto-Advance] Marking stage ${stageToView} as viewed, resetting to role selection`);
          setLastViewedStage(stageToView);
          setCurrentTurnView(0);
          setLocalSubmitted(false);
        }, 8000); // 8 seconds to view final turn
        return () => clearTimeout(timer);
      }
    }
  }, [currentTurnView, turns.length, stageToView, round]);

  // Trigger damage animation during turn stages
  useEffect(() => {
    if (isTurnStage) {
      setShowDamageAnimation(true);
      const timer = setTimeout(() => setShowDamageAnimation(false), 7000); // Show animation for 7 seconds
      return () => clearTimeout(timer);
    }
  }, [isTurnStage, currentTurnView]);

  // Countdown timer for the last 3 seconds of turn display
  useEffect(() => {
    if (isTurnStage) {
      // Start countdown at 5 seconds (showing countdown for last 3 seconds)
      const countdownStart = setTimeout(() => {
        setCountdown(3);
      }, 5000);

      return () => clearTimeout(countdownStart);
    } else {
      setCountdown(null);
    }
  }, [isTurnStage, currentTurnView]);

  // Update countdown every second
  useEffect(() => {
    if (countdown !== null && countdown > 0) {
      const timer = setTimeout(() => {
        setCountdown(countdown - 1);
      }, 1000);
      return () => clearTimeout(timer);
    }
  }, [countdown]);

  const handleRoleSelect = useCallback((role) => {
    if (!submitted && !hasTurns) {
      setSelectedRole(role);
    }
  }, [submitted, hasTurns]);

  const handleSubmit = useCallback(() => {
    if (!submitted && selectedRole !== null) {
      // Add random delay (1-10 seconds) to prevent humans from detecting bots by response time
      const randomDelay = Math.floor(Math.random() * 9000) + 1000; // 1000-10000ms
      console.log(`Adding ${randomDelay}ms delay before submitting role to mask bot response times`);

      // Store the selected role immediately on stage (not round)
      player.stage.set("selectedRole", selectedRole);

      // Set local state to immediately show waiting screen
      setLocalSubmitted(true);

      // Delay the submit to mask bot response times
      setTimeout(() => {
        player.stage.set("submit", true);
      }, randomDelay);
    }
  }, [submitted, selectedRole, player]);

  // Get virtual bots from round state (stored per-round)
  const virtualBots = round.get("virtualBots") || EMPTY_ARRAY;
  const totalPlayers = treatment?.totalPlayers;

  // Build unified player array (real + virtual)
  const allPlayers = new Array(totalPlayers).fill(null);

  // Add real players
  players.forEach(p => {
    const playerId = p.round.get("playerId");
    if (playerId !== null && playerId !== undefined) {
      allPlayers[playerId] = { type: "real", player: p, playerId };
    }
  });

  // Add virtual bots
  virtualBots.forEach(bot => {
    allPlayers[bot.playerId] = { type: "virtual", bot, playerId: bot.playerId };
  });

  // Determine which UI to show based on state
  let currentUI;
  if (isGameEndStage) {
    currentUI = 'gameEnd';
  } else if (isRoundEndStage) {
    currentUI = 'roundEnd';
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
                currentPlayerId={player.id}
                previousEnemyHealth={previousEnemyHealth}
                previousTeamHealth={previousTeamHealth}
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
                    <div className="text-gray-500 text-sm">
                      {selectedRole !== null && `Your selected role: ${["Fighter", "Tank", "Healer"][selectedRole]}`}
                    </div>
                  </div>
                )}

                {/* Turn Results - showing what happened in this turn */}
                {currentUI === 'turnResults' && (
                  <div className="w-full">
                    <ResultsPanel
                      roundNumber={roundNumber}
                      stageNumber={stageToView}
                      turnNumber={currentTurnView}
                      actions={actions}
                      allPlayers={allPlayers}
                      currentPlayerId={player.id}
                      enemyIntent={enemyIntent}
                      countdown={countdown}
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
                    />
                  </div>
                )}

                {/* Round End Screen */}
                {currentUI === 'roundEnd' && (
                  <div className="text-center">
                    <div className="text-5xl mb-4">
                      {roundOutcome === "WIN" && "🎉"}
                      {roundOutcome === "LOSE" && "💔"}
                      {roundOutcome === "TIMEOUT" && "⏰"}
                    </div>
                    <div className="text-2xl font-bold mb-3">
                      {round.get("roundEndMessage")}
                    </div>
                    <div className="text-lg text-gray-600 mb-4">
                      Round {roundNumber} complete!
                    </div>
                    {roundOutcome === "WIN" && (
                      <div className="text-xl text-green-600 font-bold mb-4">
                        +{Math.ceil(teamHealth)} Points Earned!
                      </div>
                    )}
                    <div className="text-md text-gray-500 mb-6">
                      Total Points: {totalPoints}
                    </div>
                    {submitted ? (
                      <div className="text-sm text-gray-400 mt-4">
                        <div className="text-3xl mb-2">⏳</div>
                        Waiting for next round...
                      </div>
                    ) : (
                      <button
                        onClick={() => player.stage.set("submit", true)}
                        className="bg-blue-500 hover:bg-blue-600 text-white px-8 py-4 rounded-lg text-xl font-bold transition-colors shadow-lg"
                      >
                        Continue →
                      </button>
                    )}
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

          {/* Game End Overlay */}
          {shouldShowGameEnd && (
            <div className="absolute inset-0 bg-black bg-opacity-60 flex items-center justify-center z-50">
              <GameEndScreen
                outcome={game.get("finalOutcome")}
                endMessage={stage.get("endMessage")}
                totalPoints={totalPoints}
                roundOutcomes={game.get("roundOutcomes")}
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

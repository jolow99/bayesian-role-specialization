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
  const enemyHealth = game.get("enemyHealth");
  const teamHealth = game.get("teamHealth");
  const roundNumber = round.get("roundNumber");
  const maxRounds = treatment?.maxRounds;
  const maxHealth = treatment?.maxTeamHealth;
  const maxEnemyHealth = treatment?.maxEnemyHealth;
  const currentStage = stage.get("name");
  const stageType = stage.get("stageType");
  const turnNumber = stage.get("turnNumber");

  // Check if we have valid data
  if (roundNumber === undefined || roundNumber === null) {
    return null;
  }

  // Determine which turn's data to show based on stage
  let enemyIntent, actions, damageToEnemy, damageToTeam, healAmount, previousEnemyHealth, previousTeamHealth, roles;

  if (stageType === "turn") {
    // Show data for the current turn
    enemyIntent = round.get(`turn${turnNumber}Intent`);
    actions = round.get(`turn${turnNumber}Actions`) || EMPTY_ARRAY;
    roles = round.get(`turn${turnNumber}Roles`) || EMPTY_ARRAY;
    damageToEnemy = round.get(`turn${turnNumber}DamageToEnemy`) || 0;
    damageToTeam = round.get(`turn${turnNumber}DamageToTeam`) || 0;
    healAmount = round.get(`turn${turnNumber}HealAmount`) || 0;
    previousEnemyHealth = round.get(`turn${turnNumber}PreviousEnemyHealth`) || enemyHealth;
    previousTeamHealth = round.get(`turn${turnNumber}PreviousTeamHealth`) || teamHealth;
  } else {
    // Role selection stage - no turn data yet
    enemyIntent = null;
    actions = EMPTY_ARRAY;
    roles = EMPTY_ARRAY;
    damageToEnemy = 0;
    damageToTeam = 0;
    healAmount = 0;
    previousEnemyHealth = enemyHealth;
    previousTeamHealth = teamHealth;
  }

  // Determine if this is a turn stage (showing results)
  const isTurnStage = stageType === "turn";
  const isRoleSelectionStage = stageType === "roleSelection";
  const isGameEndStage = stageType === "gameEnd";

  // Check if game has ended (either via gameEnd stage OR via outcome being set)
  const gameOutcome = game.get("outcome");
  const shouldShowGameEnd = isGameEndStage || gameOutcome;

  // Debug logging
  console.log(`[Client RENDER] Round ${roundNumber}, Stage: ${currentStage}, Type: ${stageType}, Turn: ${turnNumber}`);
  console.log(`[Client RENDER] Enemy HP: ${enemyHealth}, Team HP: ${teamHealth}`);
  console.log(`[Client RENDER] submitted: ${submitted}, isRoleSelectionStage: ${isRoleSelectionStage}, isTurnStage: ${isTurnStage}`);

  // Reset local submitted state when moving to a new round or stage type changes
  useEffect(() => {
    setLocalSubmitted(false);
  }, [roundNumber, stageType]);

  // Trigger damage animation during turn stages
  useEffect(() => {
    if (isTurnStage) {
      setShowDamageAnimation(true);
      const timer = setTimeout(() => setShowDamageAnimation(false), 7000); // Show animation for 7 seconds of the 8-second turn
      return () => clearTimeout(timer);
    }
  }, [isTurnStage, turnNumber, roundNumber]);

  // Countdown timer for the last 3 seconds of turn stage
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
  }, [isTurnStage, turnNumber, roundNumber]);

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
    if (!submitted && isRoleSelectionStage) {
      setSelectedRole(role);
    }
  }, [submitted, isRoleSelectionStage]);

  const handleSubmit = useCallback(() => {
    if (!submitted && selectedRole !== null) {
      // Add random delay (1-10 seconds) to prevent humans from detecting bots by response time
      const randomDelay = Math.floor(Math.random() * 9000) + 1000; // 1000-10000ms
      console.log(`Adding ${randomDelay}ms delay before submitting role to mask bot response times`);

      // Store the selected role immediately so UI updates
      player.round.set("selectedRole", selectedRole);

      // Set local state to immediately show waiting screen
      setLocalSubmitted(true);

      // Delay the submit to mask bot response times
      setTimeout(() => {
        player.stage.set("submit", true);
      }, randomDelay);
    }
  }, [submitted, selectedRole, player]);

  // Get virtual bots from game state
  const virtualBots = game.get("virtualBots") || EMPTY_ARRAY;
  const totalPlayers = treatment?.totalPlayers;

  // Build unified player array (real + virtual)
  const allPlayers = new Array(totalPlayers).fill(null);

  // Add real players
  players.forEach(p => {
    const playerId = p.get("playerId");
    allPlayers[playerId] = { type: "real", player: p, playerId };
  });

  // Add virtual bots
  virtualBots.forEach(bot => {
    allPlayers[bot.playerId] = { type: "virtual", bot, playerId: bot.playerId };
  });

  // Determine which UI to show based on state
  let currentUI;
  if (isGameEndStage) {
    currentUI = 'gameEnd';
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
              <h1 className="text-lg font-bold">Round {roundNumber}/{maxRounds}</h1>
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
                      turnNumber={turnNumber}
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
              </div>
            </div>
          </div>

          {/* Right Column - Battle History (full height) */}
          <div className="bg-gray-50 border-l-4 border-gray-700 overflow-hidden flex flex-col" style={{ width: '22%', minWidth: '280px', maxWidth: '350px' }}>
            <div className="bg-gray-800 text-white text-center flex-shrink-0 rounded-tr-lg flex items-center justify-center" style={{ height: '40px' }}>
              <h3 className="text-sm font-bold">
                📜 Battle History
              </h3>
            </div>
            <div className="flex-1 overflow-auto p-3 bg-white">
              <ActionHistory />
            </div>
          </div>

          {/* Game End Overlay */}
          {shouldShowGameEnd && (
            <div className="absolute inset-0 bg-black bg-opacity-60 flex items-center justify-center z-50">
              <GameEndScreen
                outcome={gameOutcome}
                endMessage={game.get("gameEndMessage") || stage.get("endMessage")}
                enemyHealth={enemyHealth}
                teamHealth={teamHealth}
                maxHealth={maxHealth}
                maxEnemyHealth={maxEnemyHealth}
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

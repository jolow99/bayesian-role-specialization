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
  const [localSubmitted, setLocalSubmitted] = useState(false); // Local state to immediately show waiting screen

  // Get current data from Empirica
  const treatment = game.get("treatment");
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
  if (isTurnStage) {
    currentUI = 'turnResults';
  } else if (submitted) {
    currentUI = 'waiting';
  } else {
    currentUI = 'roleSelection';
  }

  return (
    <div className="fixed inset-0 bg-gradient-to-b from-blue-400 to-blue-600 flex items-center justify-center p-4">
      <div className="w-full max-w-6xl" style={{ height: 'calc(100vh - 32px)' }}>
        {/* Battle Screen */}
        <div className="bg-white rounded-lg shadow-2xl border-4 border-gray-800 h-full flex flex-col">
          {/* Round Header - Fixed height */}
          <div className="bg-gray-800 text-white text-center flex-shrink-0" style={{ height: '60px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
            <h1 className="text-xl font-bold">Round {roundNumber}/{maxRounds}</h1>
            {isTurnStage && (
              <p className="text-xs text-blue-300">
                Turn {turnNumber} of 2
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
              isRevealStage={isTurnStage}
              showDamageAnimation={showDamageAnimation}
              damageToEnemy={damageToEnemy}
              damageToTeam={damageToTeam}
              healAmount={healAmount}
              actions={actions}
              allPlayers={allPlayers}
              currentPlayerId={player.id}
            />
          </div>

          {/* Role Selection or Turn Results - Fixed height */}
          <div className="bg-white border-t-4 border-gray-700 flex-shrink-0 relative" style={{ height: '240px' }}>
            <div className="p-4 h-full overflow-auto">
              {/* Waiting for other players after submitting role */}
              {currentUI === 'waiting' && (
                <div className="text-center py-6">
                  <div className="text-4xl mb-3">⏳</div>
                  <div className="text-lg font-bold text-gray-700 mb-2">Waiting for other players...</div>
                  <div className="text-gray-500 text-sm">
                    {selectedRole !== null && `Your selected role: ${["Fighter", "Tank", "Healer"][selectedRole]}`}
                  </div>
                </div>
              )}

              {/* Turn Results - showing what happened in this turn */}
              {currentUI === 'turnResults' && (
                <ResultsPanel
                  roundNumber={roundNumber}
                  turnNumber={turnNumber}
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

              {/* Role Selection Menu */}
              {currentUI === 'roleSelection' && (
                <ActionMenu
                  selectedRole={selectedRole}
                  onRoleSelect={handleRoleSelect}
                  onSubmit={handleSubmit}
                  isRoleCommitted={false}
                  currentRole={null}
                  roundsRemaining={0}
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

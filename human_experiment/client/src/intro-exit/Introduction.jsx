import React from "react";
import { MockDataProvider, TutorialWrapper } from "../components/tutorial";
import { BattleField } from "../components/BattleField";
import { ActionMenu } from "../components/ActionMenu";
import { ActionHistory } from "../components/ActionHistory";
import { ROLES } from "../constants";

// Create mock data with placeholder battle history for the tutorial
const createMockDataWithHistory = () => {
  const players = [
    { id: "player-1", playerId: 0, stats: { STR: 3, DEF: 1, SUP: 1 } },
    { id: "player-2", playerId: 1, stats: { STR: 1, DEF: 3, SUP: 1 } },
    { id: "tutorial-player", playerId: 2, stats: { STR: 1, DEF: 1, SUP: 3 } }
  ];

  // Create placeholder battle history with 1 stage and 2 turns
  const roundData = {
    roundNumber: 1,
    stageNumber: 1,
    stage1Turns: [
      {
        turnNumber: 1,
        enemyIntent: "WILL_ATTACK",
        actions: ["ATTACK", "BLOCK", "HEAL"],
        damageToEnemy: 3,
        damageToTeam: 1,
        healAmount: 3,
        previousEnemyHealth: 10,
        previousTeamHealth: 10,
        newEnemyHealth: 7,
        newTeamHealth: 10
      },
      {
        turnNumber: 2,
        enemyIntent: "WILL_REST",
        actions: ["ATTACK", "ATTACK", "ATTACK"],
        damageToEnemy: 5,
        damageToTeam: 0,
        healAmount: 0,
        previousEnemyHealth: 7,
        previousTeamHealth: 10,
        newEnemyHealth: 2,
        newTeamHealth: 10
      }
    ]
  };

  return {
    game: {
      enemyHealth: 2,
      maxEnemyHealth: 10,
      teamHealth: 10,
      maxTeamHealth: 10,
      treatment: {
        totalPlayers: 3,
        maxRounds: 5,
        maxEnemyHealth: 10,
        maxTeamHealth: 10
      }
    },
    player: {
      id: "tutorial-player",
      playerId: 2,
      stats: { STR: 1, DEF: 1, SUP: 3 },
      roleOrder: [ROLES.FIGHTER, ROLES.TANK, ROLES.MEDIC],
      stage: {},
      round: {},
      get: (key) => {
        if (key === "actionHistory") return [{ stage: 1, role: "MEDIC" }];
        return undefined;
      }
    },
    players: players,
    round: {
      ...roundData,
      get: (key) => roundData[key]
    },
    stage: {
      name: "roleSelection",
      stageType: "roleSelection"
    }
  };
};

export function Introduction({ next }) {
  // Create mock data for the tutorial with placeholder battle history
  const mockData = createMockDataWithHistory();

  // Define the tutorial steps
  const tutorialSteps = [
    {
      targetId: "full-screen", // Show darkened background
      tooltipPosition: "center",
      showBorder: false, // Don't show blue pulsing border
      content: (
        <div>
          <h3 className="text-lg font-bold text-gray-900 mb-3">Welcome!</h3>
          <p className="text-sm text-gray-700 mb-3">
            You'll play <strong>8 rounds</strong> of this cooperative game. In each round, work with your team to defeat the enemy before your team's health reaches 0.
          </p>
          <p className="text-sm text-gray-700">
            Let's take a tour of the interface to understand how to play!
          </p>
        </div>
      )
    },
    {
      targetId: "battlefield",
      tooltipPosition: "right",
      content: (
        <div>
          <h4 className="text-lg font-bold text-gray-900 mb-2">The Battlefield</h4>
          <p className="text-sm text-gray-700 mb-2">
          <ul className="text-sm text-gray-700 space-y-2 ml-4">
            <li>• On the left is your team. You are playing with 2 other players who are either humans or bots. Your team's health is the green bar.</li>
            <li>• On the right is the enemy, whose health is the red bar. </li>
          </ul>
          </p>
          <p className="text-sm text-gray-700">
            Your goal is to reduce the enemy's health to 0 before your team's health reaches 0.
          </p>
        </div>
      )
    },
    {
      targetId: "action-menu",
      tooltipPosition: "top",
      content: (
        <div>
          <h4 className="text-lg font-bold text-gray-900 mb-2">Role Selection</h4>
          <p className="text-sm text-gray-700 mb-3">
            At each stage you'll have to choose a role, which determines the actions your character takes. 
            You'll be committed to the role for 2 turns:
          </p>
          <ul className="text-sm text-gray-700 space-y-2 ml-4">
            <li><strong>🤺 Fighter:</strong> Attacks most of the time</li>
            <li><strong>💂 Tank:</strong> Blocks most of the time if the enemy is attacking. Otherwise, acts like a fighter</li>
            <li><strong>👩🏻‍⚕️ Medic:</strong> Heals most of the time if the team's health is not full. Otherwise, acts like a fighter</li>
          </ul>
        </div>
      )
    },
    {
      targetId: "player-stats",
      tooltipPosition: "right",
      content: (
        <div>
          <h4 className="text-lg font-bold text-gray-900 mb-2">Player Stats & How They Combine</h4>
          <p className="text-sm text-gray-700 mb-3">
            Each player has unique stats that determine how effective they are at each action:
          </p>
          <div className="space-y-2 mb-3">
            <div>
              <div className="font-semibold text-red-600 mb-1">⚔️ STR (Strength) → Attack</div>
              <div className="text-sm text-gray-700">Damage to enemy = <strong>sum</strong> of all attackers' STR</div>
            </div>
            <div>
              <div className="font-semibold text-blue-600 mb-1">🛡️ DEF (Defense) → Block</div>
              <div className="text-sm text-gray-700">Damage blocked = <strong>highest</strong> DEF among blockers</div>
            </div>
            <div>
              <div className="font-semibold text-green-600 mb-1">💚 SUP (Support) → Heal</div>
              <div className="text-sm text-gray-700">Health restored = <strong>sum</strong> of all healers' SUP</div>
            </div>
          </div>
          <p className="text-sm text-gray-700 italic">
            Tip: You can hover over the ⓘ button anytime during the game to review this!
          </p>
        </div>
      )
    },
    {
      targetId: "battle-history",
      tooltipPosition: "left",
      content: (
        <div>
          <h4 className="text-lg font-bold text-gray-900 mb-2">Battle History</h4>
          <p className="text-sm text-gray-700 mb-2">
            The battle history shows past stages and turns. This helps you coordinate with your team by understanding their strategies.
          </p>
          <p className="text-sm text-gray-700">
            Let's look at how to read it in detail...
          </p>
        </div>
      )
    },
    {
      targetId: "battle-history-s1t1",
      tooltipPosition: "left",
      content: (
        <div>
          <h4 className="text-lg font-bold text-gray-900 mb-2">Reading Turn Details</h4>
          <p className="text-sm text-gray-700 mb-2">
            Each turn entry shows:
          </p>
          <ul className="text-sm text-gray-700 space-y-1 ml-4 mb-2">
            <li>• <strong>Turn number</strong> and whether the <strong>enemy attacked or rested</strong></li>
            <li>• <strong>Health status</strong> after the turn (👥 Team, 👹 Enemy)</li>
            <li>• <strong>Action icons</strong> for each player (⚔️ Attack, 🛡️ Block, 💚 Heal)</li>
            <li>• <strong>"YOU"</strong> label marks your own actions</li>
          </ul>
          <p className="text-sm text-gray-700 italic">
            In this example: P1 attacked, P2 blocked, and YOU healed. The enemy attacked but only dealt 1 damage thanks to P2's block!
          </p>
        </div>
      )
    },
    {
      targetId: "full-screen",
      tooltipPosition: "center",
      showBorder: false,
      content: (
        <div>
          <h4 className="text-lg font-bold text-gray-900 mb-2">Points & Bonus</h4>
          <p className="text-sm text-gray-700 mb-3">
            You earn points based on how quickly you win each round. Faster victories earn more points!
          </p>
          <ul className="text-sm text-gray-700 space-y-1 ml-4 mb-3">
            <li>• Each round has a maximum of <strong>10 turns</strong></li>
            <li>• <strong>Win up to 100 points</strong> - for every additional turn you take, you get 10 points less</li>
            <li>• <strong>Lose or timeout:</strong> Earn 0 points</li>
          </ul>
          <p className="text-sm text-gray-700 mb-3">
            <strong>Bonus Payment:</strong> Every 40 points = $0.10 bonus (up to ~$1.00 total bonus on top of base payment)
          </p>
          <p className="text-sm text-gray-700 italic">
            You're now ready for a trial game!
          </p>
        </div>
      )
    }
  ];

  const handleComplete = () => {
    if (next) next();
  };

  // Build allPlayers array for BattleField component
  // Player at index 2 is the tutorial player (real), others are virtual
  const allPlayers = mockData.players.map((p, idx) => ({
    type: idx === 2 ? "real" : "virtual",
    player: p,
    playerId: p.playerId,
    bot: idx === 2 ? null : { stats: p.stats, playerId: p.playerId }
  }));

  return (
    <MockDataProvider mockData={mockData}>
      <TutorialWrapper steps={tutorialSteps} onComplete={handleComplete}>
        <div className="fixed inset-0 bg-gradient-to-b from-blue-400 to-blue-600 flex items-center justify-center p-2" data-tutorial-id="full-screen">
          <div className="w-full h-full flex items-center justify-center" style={{ maxWidth: '1400px' }}>
            {/* Battle Screen */}
            <div className="bg-white rounded-lg shadow-2xl border-4 border-gray-800 w-full h-full flex overflow-hidden relative">
              {/* Left Column - Game Interface */}
              <div className="flex-1 flex flex-col min-w-0">
                {/* Round Header */}
                <div className="bg-gray-800 text-white text-center flex-shrink-0 rounded-tl-lg flex items-center justify-center" style={{ height: '40px' }}>
                  <h1 className="text-lg font-bold">Round 1/5</h1>
                </div>

                {/* Battle Field */}
                <div className="flex-shrink-0" style={{ height: '35vh', minHeight: '250px', maxHeight: '400px' }}>
                  <BattleField
                    enemyHealth={mockData.game.enemyHealth}
                    maxEnemyHealth={mockData.game.maxEnemyHealth}
                    teamHealth={mockData.game.teamHealth}
                    maxHealth={mockData.game.maxTeamHealth}
                    enemyIntent={null}
                    isRevealStage={false}
                    showDamageAnimation={false}
                    damageToEnemy={0}
                    damageToTeam={0}
                    healAmount={0}
                    actions={[]}
                    allPlayers={allPlayers}
                    currentPlayerGameId={2}
                    previousEnemyHealth={mockData.game.enemyHealth}
                    previousTeamHealth={mockData.game.teamHealth}
                    bossDamage={3}
                    enemyAttackProbability={1.0}
                  />
                </div>

                {/* Role Selection */}
                <div className="bg-white border-t-4 border-gray-700 flex-1 min-h-0 flex flex-col">
                  <div className="flex-1 p-4 flex items-center justify-center">
                    <div className="w-full max-w-4xl">
                      <ActionMenu
                        selectedRole={null}
                        onRoleSelect={() => {}}
                        onSubmit={() => {}}
                        isRoleCommitted={false}
                        currentRole={null}
                        roundsRemaining={0}
                        submitted={false}
                        roleOrder={[ROLES.FIGHTER, ROLES.TANK, ROLES.MEDIC]}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Right Column - Battle History */}
              <div className="bg-gray-50 border-l-4 border-gray-700 overflow-hidden flex flex-col" style={{ width: '22%', minWidth: '280px', maxWidth: '350px' }} data-tutorial-id="battle-history">
                <div className="bg-gray-800 text-white text-center flex-shrink-0 rounded-tr-lg flex items-center justify-center" style={{ height: '40px' }}>
                  <h3 className="text-sm font-bold">
                    📜 Battle History
                  </h3>
                </div>
                <div className="flex-1 overflow-auto p-3 bg-white">
                  <ActionHistory />
                </div>
              </div>
            </div>
          </div>
        </div>
      </TutorialWrapper>
    </MockDataProvider>
  );
}

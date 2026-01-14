import React from "react";
import { MockDataProvider, TutorialWrapper, createDefaultMockData } from "../components/tutorial";
import { BattleField } from "../components/BattleField";
import { ActionMenu } from "../components/ActionMenu";
import { ActionHistory } from "../components/ActionHistory";
import { ROLES } from "../constants";

export function Introduction({ next }) {
  // Create mock data for the tutorial
  const mockData = createDefaultMockData();

  // Define the 7 tutorial steps
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
          <p className="text-sm text-gray-700 mb-3">
            <strong>Points & Bonus:</strong> You earn points based on how quickly you win each round. Faster victories earn more points!
          </p>
          <ul className="text-sm text-gray-700 space-y-1 ml-4 mb-3">
            <li>• Each round has a maximum of <strong>10 turns</strong></li>
            <li>• <strong>Win up to 100 points</strong> - for every additional turn you take, you get 10 points less</li>
            <li>• <strong>Lose or timeout:</strong> Earn 0 points</li>
          </ul>
          <p className="text-sm text-gray-700 mb-2">
            <strong>Bonus Payment:</strong> Every 40 points = $0.10 bonus (up to ~$1.00 total bonus on top of base payment)
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
            The battlefield shows the enemy's health (red bar) and your team's health (green bar).
          </p>
          <p className="text-sm text-gray-700">
            Your goal is to reduce the enemy's health to 0 before your team's health reaches 0.
          </p>
        </div>
      )
    },
    {
      targetId: "player-stats",
      tooltipPosition: "right",
      content: (
        <div>
          <h4 className="text-lg font-bold text-gray-900 mb-2">Player Stats</h4>
          <p className="text-sm text-gray-700 mb-3">
            Each player has unique stats that determine how effective they are at each action:
          </p>
          <ul className="text-sm text-gray-700 space-y-2 ml-4">
            <li><strong className="text-red-600">STR (Strength):</strong> Determines attack damage</li>
            <li><strong className="text-blue-600">DEF (Defense):</strong> Determines damage blocked when blocking</li>
            <li><strong className="text-green-600">SUP (Support):</strong> Determines healing effectiveness</li>
          </ul>
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
            At the start, you'll choose one of three roles. Your role influences the actions taken and
            you'll be committed to the role for 2 turns:
          </p>
          <ul className="text-sm text-gray-700 space-y-2 ml-4">
            <li><strong>🤺 Fighter:</strong> Attacks most of the time</li>
            <li><strong>💂 Tank:</strong> Blocks most of the time if the enemy is attacking. Otherwise, acts like a fighter</li>
            <li><strong>👩🏻‍⚕️ Medic:</strong> Heals most of the time if the team's health is less than or equal to 50%. Otherwise, acts like a fighter</li>
          </ul>
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
            The battle history shows past stages and turns. You can see what actions each player took and the results.
          </p>
          <p className="text-sm text-gray-700">
            This helps you coordinate with your team by understanding their strategies.
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
          <h4 className="text-lg font-bold text-gray-900 mb-2">Turn Counter & Time Pressure</h4>
          <p className="text-sm text-gray-700 mb-3">
            At the top of the screen, you'll see a turn counter showing <strong>Turn: X / 10</strong>
          </p>
          <p className="text-sm text-gray-700 mb-3">
            Remember: The faster you defeat the enemy, the more points you earn!
          </p>
          <ul className="text-sm text-gray-700 space-y-2 ml-4">
            <li>• Win in 1-2 turns: Earn 80-90 points</li>
            <li>• Win in 5 turns: Earn 50 points</li>
            <li>• Win in 10 turns: Earn 0 points</li>
            <li>• Lose or exceed 10 turns: Earn 0 points</li>
          </ul>
        </div>
      )
    },
    {
      targetId: "stats-info",
      tooltipPosition: "bottom",
      content: (
        <div>
          <h4 className="text-lg font-bold text-gray-900 mb-2">How Stats Combine</h4>
          <p className="text-sm text-gray-700 mb-3">
            Hover over this info button anytime during the game to see how stats combine:
          </p>
          <div className="space-y-2 mb-3">
            <div>
              <div className="font-semibold text-red-600 mb-1">⚔️ Attack</div>
              <div className="text-sm text-gray-700">The amount of damage taken by a boss is the sum of all STR stats of players who attack</div>
            </div>
            <div>
              <div className="font-semibold text-blue-600 mb-1">🛡️ Block</div>
              <div className="text-sm text-gray-700">The amount of damage blocked by the team is the highest DEF stat amongst players who defend</div>
            </div>
            <div>
              <div className="font-semibold text-green-600 mb-1">💚 Heal</div>
              <div className="text-sm text-gray-700">The amount of health healed by the team is the sum of all SUP stats of players who heal</div>
            </div>
          </div>
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
  const allPlayers = mockData.players.map((p, idx) => ({
    type: idx === 0 ? "real" : "virtual",
    player: p,
    playerId: p.playerId,
    bot: idx === 0 ? null : { stats: p.stats, playerId: p.playerId }
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
                    currentPlayerId="tutorial-player"
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

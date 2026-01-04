import React from "react";
import { MockDataProvider, TutorialWrapper, createDefaultMockData } from "../components/tutorial";
import { BattleField } from "../components/BattleField";
import { ActionMenu } from "../components/ActionMenu";
import { ActionHistory } from "../components/ActionHistory";

const ROLES = { FIGHTER: 0, TANK: 1, HEALER: 2 };

export function Introduction({ next, game }) {
  // Create mock data for the tutorial
  const mockData = createDefaultMockData();

  // Define the 6 tutorial steps
  const tutorialSteps = [
    {
      targetId: null, // Highlight entire interface
      tooltipPosition: "center",
      content: (
        <div>
          <h3 className="text-lg font-bold text-gray-900 mb-3">Welcome to Cooperative Battle Game!</h3>
          <p className="text-sm text-gray-700 mb-2">
            This is the main battle screen. You'll see the enemy at the top right, your team at the bottom left,
            and the battle history on the right side.
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
            <li><strong className="text-blue-600">DEF (Defense):</strong> Determines damage blocked when defending</li>
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
            At the start, you'll choose one of three roles. Your role determines your action and
            you'll be committed to the role for 2 rounds:
          </p>
          <ul className="text-sm text-gray-700 space-y-2 ml-4">
            <li><strong>⚔️ Fighter:</strong> Attacks most of the time</li>
            <li><strong>🛡️ Tank:</strong> Defends when the enemy is attacking, most of the time. Otherwise, acts like a fighter</li>
            <li><strong>💚 Healer:</strong> Heals if the team's health is less than 50%, most of the time. Otherwise, acts like a fighter</li>
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
            The battle history shows past rounds and turns. You can see what actions each player took and the results.
          </p>
          <p className="text-sm text-gray-700">
            This helps you coordinate with your team by understanding their strategies.
          </p>
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
            Click this info button anytime to learn how stats combine:
          </p>
          <ul className="text-sm text-gray-700 space-y-2 ml-4">
            <li><strong>Attack damage:</strong> All attackers' STR stats are added together</li>
            <li><strong>Defense:</strong> Only the highest DEF stat among defenders counts (not the sum)</li>
            <li><strong>Healing:</strong> All healers' SUP stats are added together (up to max team health)</li>
          </ul>
          <p className="text-sm text-gray-700 mt-3 italic">
            You're now ready for the tutorial! Click "Next" to start.
          </p>
        </div>
      )
    }
  ];

  const handleComplete = () => {
    if (next) next();
  };

  const handleSkip = () => {
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
      <TutorialWrapper steps={tutorialSteps} onComplete={handleComplete} onSkip={handleSkip}>
        <div className="fixed inset-0 bg-gradient-to-b from-blue-400 to-blue-600 flex items-center justify-center p-2">
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
                        roleOrder={[ROLES.FIGHTER, ROLES.TANK, ROLES.HEALER]}
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

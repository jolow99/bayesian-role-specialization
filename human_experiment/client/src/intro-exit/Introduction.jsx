import React from "react";
import { Button } from "../components/Button";

export function Introduction({ next, game }) {
  return (
    <div className="mt-3 sm:mt-5 p-20 max-w-4xl mx-auto">
      <h3 className="text-3xl leading-6 font-bold text-gray-900 mb-6 text-center">
        Cooperative Battle Game
      </h3>

      <div className="space-y-6">
        {/* Game Overview */}
        <section>
          <h4 className="text-lg font-semibold text-gray-800 mb-3">Game Overview</h4>
          <p className="text-sm text-gray-700 mb-2">
            You and 2 other players will work together as a team to defeat a powerful enemy.
          </p>
          <p className="text-sm text-gray-700">
            Your goal is to reduce the enemy's health to 0 before your team's health reaches 0.
          </p>
        </section>

        {/* Actions */}
        <section>
          <h4 className="text-lg font-semibold text-gray-800 mb-3">Actions</h4>
          <p className="text-sm text-gray-700 mb-3">
            Each round, you'll choose one of three actions:
          </p>
          <div className="space-y-2 bg-gray-50 p-4 rounded">
            <div className="flex items-start gap-2">
              <span className="text-2xl">⚔️</span>
              <div>
                <strong>Attack:</strong> Deal damage to the enemy
              </div>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-2xl">🛡️</span>
              <div>
                <strong>Defend:</strong> Protect your team from enemy attacks
              </div>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-2xl">💚</span>
              <div>
                <strong>Heal:</strong> Restore your team's health
              </div>
            </div>
          </div>
        </section>

        {/* Player Stats */}
        <section>
          <h4 className="text-lg font-semibold text-gray-800 mb-3">Player Stats</h4>
          <p className="text-sm text-gray-700 mb-3">
            Each player has unique stats that determine how effective they are at each action:
          </p>
          <div className="space-y-2 bg-gray-50 p-4 rounded">
            <div className="flex items-start gap-2">
              <span className="text-xl">⚔️</span>
              <div>
                <strong>Strength (STR):</strong> Determines attack damage
              </div>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-xl">🛡️</span>
              <div>
                <strong>Defense (DEF):</strong> Determines damage blocked when defending
              </div>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-xl">💚</span>
              <div>
                <strong>Support (SUP):</strong> Determines healing effectiveness
              </div>
            </div>
          </div>
          <div className="bg-blue-50 border border-blue-200 rounded p-3 mt-3">
            <p className="text-sm text-blue-800">
              <strong>Note:</strong> When multiple players defend, only the highest defense stat counts (not the sum).
            </p>
          </div>
        </section>

        {/* Game Mechanics */}
        <section>
          <h4 className="text-lg font-semibold text-gray-800 mb-3">Game Mechanics</h4>
          <div className="space-y-3">
            <div className="bg-green-50 border border-green-300 rounded p-3">
              <div className="font-bold text-green-700 mb-1">Victory</div>
              <p className="text-sm text-gray-700">
                Reduce enemy health to 0
              </p>
            </div>
            <div className="bg-red-50 border border-red-300 rounded p-3">
              <div className="font-bold text-red-700 mb-1">Defeat</div>
              <p className="text-sm text-gray-700">
                Team health reaches 0
              </p>
            </div>
          </div>
          <p className="text-sm text-gray-700 mt-3">
            The enemy will randomly decide each turn whether to attack or rest. When the enemy intends to attack, you'll see a warning.
          </p>
        </section>
      </div>

      <div className="flex justify-center mt-8">
        <Button handleClick={next} autoFocus>
          <p>Start Game</p>
        </Button>
      </div>
    </div>
  );
}

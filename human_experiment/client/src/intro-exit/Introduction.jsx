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

        {/* Roles */}
        <section>
          <h4 className="text-lg font-semibold text-gray-800 mb-3">Roles</h4>
          <p className="text-sm text-gray-700 mb-3">
            At the start, you'll choose one of three roles. Your role determines your action and you'll be committed to the role for 2 rounds after which you can select your role again:
          </p>
          <div className="space-y-2 bg-gray-50 p-4 rounded">
            <div className="flex items-start gap-2">
              <span className="text-2xl">⚔️</span>
              <div>
                <strong>Fighter:</strong> Attacks most of the time.
              </div>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-2xl">🛡️</span>
              <div>
                <strong>Tank:</strong> Defends when the enemy is attacking, most of the time. Otherwise, acts like a fighter
              </div>
            </div>
            <div className="flex items-start gap-2">
              <span className="text-2xl">💚</span>
              <div>
                <strong>Healer:</strong> Heals if the team's health is less than 50%, most of the time. Otherwise, acts like a fighter.
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
              <div>
                <strong>Strength (STR):</strong> Determines attack damage
              </div>
            </div>
            <div className="flex items-start gap-2">
              <div>
                <strong>Defense (DEF):</strong> Determines damage blocked when defending
              </div>
            </div>
            <div className="flex items-start gap-2">
              <div>
                <strong>Support (SUP):</strong> Determines healing effectiveness
              </div>
            </div>
          </div>
          <div className="bg-blue-50 border border-blue-200 rounded p-3 mt-3">
            <p className="text-sm text-blue-800 mb-2">
              <strong>How stats combine:</strong>
            </p>
            <ul className="text-sm text-blue-800 space-y-1 ml-4 list-disc">
              <li><strong>Attack damage:</strong> All attackers' STR stats are added together</li>
              <li><strong>Defense:</strong> Only the highest DEF stat among defenders counts (not the sum)</li>
              <li><strong>Healing:</strong> All healers' SUP stats are added together (up to max team health)</li>
            </ul>
          </div>
        </section>

        {/* Tutorial Introduction */}
        <section className="bg-yellow-50 border border-yellow-300 rounded-lg p-4">
          <h4 className="text-lg font-semibold text-gray-800 mb-3">Tutorial</h4>
          <p className="text-sm text-gray-700 mb-3">
            Before we start the main game, you'll play through a short tutorial
            to familiarize yourself with the game mechanics.
          </p>
        </section>

      </div>

      <div className="flex justify-center mt-8">
        <Button handleClick={next} autoFocus>
          <p>Start Tutorial</p>
        </Button>
      </div>
    </div>
  );
}

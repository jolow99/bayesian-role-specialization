import React from "react";
import { usePlayer, useStage } from "@empirica/core/player/classic/react";

export const GameEndScreen = React.memo(function GameEndScreen({
  outcome,
  endMessage,
  enemyHealth,
  teamHealth,
  maxHealth,
  maxEnemyHealth
}) {
  const player = usePlayer();
  const stage = useStage();
  const submitted = player.stage.get("submit");
  const stageType = stage.get("stageType");

  const handleContinue = () => {
    player.stage.set("submit", true);
  };

  // If we're showing the overlay during a turn stage (game just ended),
  // don't show the button yet - wait for the actual gameEnd stage
  const isActualGameEndStage = stageType === "gameEnd";

  // Determine outcome styling and icon
  let bgColorClass, borderColorClass, textColorClass, icon, title, message;

  if (outcome === "WIN") {
    bgColorClass = "bg-green-50";
    borderColorClass = "border-green-400";
    textColorClass = "text-green-700";
    icon = "🎉";
    title = "Victory!";
    message = "You defeated the enemy!";
  } else if (outcome === "LOSE") {
    bgColorClass = "bg-red-50";
    borderColorClass = "border-red-400";
    textColorClass = "text-red-700";
    icon = "💀";
    title = "Defeat";
    message = "Your team was defeated.";
  } else {
    bgColorClass = "bg-yellow-50";
    borderColorClass = "border-yellow-400";
    textColorClass = "text-yellow-700";
    icon = "⏱️";
    title = "Time's Up!";
    message = "The battle has ended.";
  }

  return (
    <div className="flex items-center justify-center h-full">
      <div className={`${bgColorClass} border-4 ${borderColorClass} rounded-xl p-8 max-w-2xl w-full shadow-2xl`}>
        {/* Icon and Title */}
        <div className="text-center mb-6">
          <div className="text-8xl mb-4">{icon}</div>
          <h1 className={`text-5xl font-bold ${textColorClass} mb-2`}>{title}</h1>
          <p className="text-xl text-gray-700">{message}</p>
        </div>

        {/* Final Stats */}
        <div className="bg-white rounded-lg p-6 mb-6 border-2 border-gray-300">
          <h3 className="text-lg font-bold text-gray-800 mb-4 text-center">Final Battle Statistics</h3>

          <div className="grid grid-cols-2 gap-4">
            {/* Enemy Health */}
            <div className="text-center">
              <div className="text-sm text-gray-600 mb-2">Enemy Health</div>
              <div className="flex items-center justify-center gap-2">
                <div className="text-3xl">👹</div>
                <div className="text-2xl font-bold text-red-600">
                  {enemyHealth} / {maxEnemyHealth}
                </div>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3 mt-2">
                <div
                  className="bg-red-500 h-3 rounded-full transition-all duration-300"
                  style={{ width: `${Math.max(0, (enemyHealth / maxEnemyHealth) * 100)}%` }}
                />
              </div>
            </div>

            {/* Team Health */}
            <div className="text-center">
              <div className="text-sm text-gray-600 mb-2">Team Health</div>
              <div className="flex items-center justify-center gap-2">
                <div className="text-3xl">❤️</div>
                <div className="text-2xl font-bold text-green-600">
                  {teamHealth} / {maxHealth}
                </div>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-3 mt-2">
                <div
                  className="bg-green-500 h-3 rounded-full transition-all duration-300"
                  style={{ width: `${Math.max(0, (teamHealth / maxHealth) * 100)}%` }}
                />
              </div>
            </div>
          </div>
        </div>

        {/* Next Button */}
        <div className="text-center">
          {!isActualGameEndStage ? (
            <div className="text-gray-600 text-lg">
              <div className="text-3xl mb-2">⏳</div>
              Waiting for round to complete...
            </div>
          ) : submitted ? (
            <div className="text-gray-600 text-lg">
              <div className="text-3xl mb-2">⏳</div>
              Waiting for other players...
            </div>
          ) : (
            <button
              onClick={handleContinue}
              className={`${textColorClass} ${bgColorClass} border-2 ${borderColorClass} px-8 py-4 rounded-lg text-xl font-bold hover:opacity-80 transition-opacity shadow-lg`}
            >
              Next →
            </button>
          )}
        </div>
      </div>
    </div>
  );
});

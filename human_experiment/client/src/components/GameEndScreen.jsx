import React from "react";
import { usePlayer, useStage } from "@empirica/core/player/classic/react";

export const GameEndScreen = React.memo(function GameEndScreen({
  outcome,
  endMessage,
  totalPoints,
  roundOutcomes = []
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

  // Calculate wins/losses/timeouts
  const wins = roundOutcomes.filter(r => r.outcome === "WIN").length;
  const losses = roundOutcomes.filter(r => r.outcome === "LOSE").length;
  const timeouts = roundOutcomes.filter(r => r.outcome === "TIMEOUT").length;

  // Determine overall outcome styling
  const bgColorClass = "bg-blue-50";
  const borderColorClass = "border-blue-400";
  const textColorClass = "text-blue-700";
  const icon = "🎮";
  const title = "Game Complete!";
  const message = endMessage || `Finished all rounds!`;

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
          <h3 className="text-lg font-bold text-gray-800 mb-4 text-center">Game Summary</h3>

          {/* Total Points */}
          <div className="text-center mb-6">
            <div className="text-5xl font-bold text-blue-600 mb-2">
              {totalPoints || 0}
            </div>
            <div className="text-lg text-gray-600">Total Points Earned</div>
          </div>

          {/* Win/Loss Stats */}
          <div className="grid grid-cols-3 gap-4 mb-4">
            <div className="text-center">
              <div className="text-3xl mb-1">🎉</div>
              <div className="text-2xl font-bold text-green-600">{wins}</div>
              <div className="text-sm text-gray-600">Wins</div>
            </div>
            <div className="text-center">
              <div className="text-3xl mb-1">💔</div>
              <div className="text-2xl font-bold text-red-600">{losses}</div>
              <div className="text-sm text-gray-600">Losses</div>
            </div>
            <div className="text-center">
              <div className="text-3xl mb-1">⏰</div>
              <div className="text-2xl font-bold text-yellow-600">{timeouts}</div>
              <div className="text-sm text-gray-600">Timeouts</div>
            </div>
          </div>

          {/* Round Details */}
          <div className="border-t border-gray-200 pt-4">
            <div className="text-xs font-semibold text-gray-600 mb-2 uppercase tracking-wide text-center">
              Round Results
            </div>
            <div className="flex flex-wrap justify-center gap-2">
              {roundOutcomes.map((round, idx) => (
                <div key={idx} className="text-center">
                  <div className="text-sm">
                    {round.outcome === "WIN" && "🎉"}
                    {round.outcome === "LOSE" && "💔"}
                    {round.outcome === "TIMEOUT" && "⏰"}
                  </div>
                  <div className="text-xs text-gray-500">R{round.roundNumber}</div>
                </div>
              ))}
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

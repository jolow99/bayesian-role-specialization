import React, { useState } from "react";
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
  const [localSubmitted, setLocalSubmitted] = useState(false);

  const handleContinue = () => {
    if (localSubmitted) return; // Prevent double-click

    const isRoundEnd = stageType === "roundEnd";

    if (isRoundEnd) {
      // Add random delay (1-10 seconds) for round transitions to mask bot/human differences
      const randomDelay = Math.floor(Math.random() * 9000) + 1000; // 1000-10000ms
      console.log(`Adding ${randomDelay}ms delay before submitting round end to mask bot response times`);

      // Set local state to immediately show waiting screen
      setLocalSubmitted(true);

      // Delay the submit to mask bot response times
      setTimeout(() => {
        player.stage.set("submit", true);
      }, randomDelay);
    } else {
      // Game end - no delay needed
      player.stage.set("submit", true);
    }
  };

  // If we're showing the overlay during a turn stage (game just ended),
  // don't show the button yet - wait for the actual gameEnd or roundEnd stage
  const isActualEndStage = stageType === "gameEnd" || stageType === "roundEnd";
  const isGameEnd = stageType === "gameEnd";
  const isRoundEnd = stageType === "roundEnd";

  // Calculate wins/losses/timeouts
  const wins = roundOutcomes.filter(r => r.outcome === "WIN").length;
  const losses = roundOutcomes.filter(r => r.outcome === "LOSE").length;
  const timeouts = roundOutcomes.filter(r => r.outcome === "TIMEOUT").length;

  // Determine overall outcome styling based on outcome or stage type
  let bgColorClass, borderColorClass, textColorClass, icon, title, message;

  if (isRoundEnd) {
    // Round end - style based on outcome
    if (outcome === "WIN") {
      bgColorClass = "bg-green-50";
      borderColorClass = "border-green-400";
      textColorClass = "text-green-700";
      icon = "🎉";
      title = "Victory!";
    } else if (outcome === "LOSE") {
      bgColorClass = "bg-red-50";
      borderColorClass = "border-red-400";
      textColorClass = "text-red-700";
      icon = "💔";
      title = "Defeat";
    } else {
      bgColorClass = "bg-yellow-50";
      borderColorClass = "border-yellow-400";
      textColorClass = "text-yellow-700";
      icon = "⏰";
      title = "Time's Up!";
    }
    message = endMessage || "Round complete!";
  } else {
    // Game end - blue styling
    bgColorClass = "bg-blue-50";
    borderColorClass = "border-blue-400";
    textColorClass = "text-blue-700";
    icon = "🎮";
    title = "Game Complete!";
    message = endMessage || `Finished all rounds!`;
  }

  return (
    <div className="flex items-center justify-center h-full px-4">
      <div className={`${bgColorClass} border-4 ${borderColorClass} rounded-xl p-8 max-w-4xl w-full shadow-2xl`}>
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
          {!isActualEndStage ? (
            <div className="text-gray-600 text-lg">
              <div className="text-3xl mb-2">⏳</div>
              Waiting for round to complete...
            </div>
          ) : (submitted || localSubmitted) ? (
            <div className="text-gray-600 text-lg">
              <div className="text-3xl mb-2">⏳</div>
              {isRoundEnd ? "Waiting for next round..." : "Waiting for other players..."}
            </div>
          ) : (
            <button
              onClick={handleContinue}
              className={`${textColorClass} ${bgColorClass} border-2 ${borderColorClass} px-8 py-4 rounded-lg text-xl font-bold hover:opacity-80 transition-opacity shadow-lg`}
            >
              {isRoundEnd ? "Go to Next Round →" : "Next →"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
});

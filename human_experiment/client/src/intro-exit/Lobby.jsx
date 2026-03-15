import React from "react";
import { usePlayer } from "@empirica/core/player/classic/react";

export function Lobby() {
  const player = usePlayer();

  // Get lobby counts from server-broadcast attributes
  const connectedCount = player?.get("lobbyPlayersConnected") || 1; // At least we're connected
  const totalRequired = player?.get("lobbyPlayersRequired") || player?.get("treatment")?.playerCount || 3;

  // Build player status array based on connection count
  const playerSlots = Array(totalRequired).fill(null).map((_, idx) => ({
    position: idx + 1,
    connected: idx < connectedCount,
  }));

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-400 to-blue-600 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-2xl border-4 border-gray-800 p-8 max-w-lg w-full">
        <div className="text-center">
          <div className="text-6xl mb-4">⏳</div>
          <h1 className="text-2xl font-bold text-gray-800 mb-4">
            Waiting for Players
          </h1>

          <p className="text-gray-600 mb-6">
            Please wait while we match you with other participants.
            This may take up to <span className="font-semibold">15 minutes</span>.
          </p>

          {/* Player connection status */}
          <div className="bg-gray-100 rounded-lg p-4 mb-6">
            <div className="text-sm text-gray-500 mb-3">Player Status</div>
            <div className="flex justify-center gap-6">
              {playerSlots.map((slot) => (
                <div key={slot.position} className="flex flex-col items-center">
                  <div className={`w-12 h-12 rounded-full flex items-center justify-center text-lg font-bold ${
                    slot.connected
                      ? "bg-green-500 text-white"
                      : "bg-gray-300 text-gray-500 animate-pulse"
                  }`}>
                    P{slot.position}
                  </div>
                  <span className={`text-xs mt-1 ${
                    slot.connected ? "text-green-600" : "text-gray-400"
                  }`}>
                    {slot.connected ? "Ready" : "Waiting..."}
                  </span>
                </div>
              ))}
            </div>
            <div className="text-sm text-gray-600 mt-3">
              {connectedCount} of {totalRequired} players connected
            </div>
          </div>

          <div className="text-sm text-gray-500">
            <p className="mb-2">
              Please keep this tab open and active.
            </p>
            <p>
              The game will start automatically once all players are ready.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

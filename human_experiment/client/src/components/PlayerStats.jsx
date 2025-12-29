import React from "react";

export function PlayerStats({ stats, showStats, playerName = "You" }) {
  if (!showStats || !stats) {
    return null;
  }

  const statConfig = [
    { key: "STR", label: "Strength", color: "bg-red-500", icon: "⚔️" },
    { key: "DEF", label: "Defense", color: "bg-blue-500", icon: "🛡️" },
    { key: "SUP", label: "Support", color: "bg-green-500", icon: "💚" }
  ];

  // Stats sum to 6, calculate percentage for display bar
  const totalStats = 6;

  return (
    <div className="bg-white rounded-lg border border-gray-300 p-4">
      <h3 className="text-sm font-semibold text-gray-700 mb-3">
        {playerName}'s Stats
      </h3>
      <div className="space-y-2">
        {statConfig.map(({ key, label, color, icon }) => {
          const value = stats[key] || 0;
          const percentage = Math.round((value / totalStats) * 100);

          return (
            <div key={key} className="flex items-center gap-2">
              <span className="text-lg">{icon}</span>
              <div className="flex-1">
                <div className="flex justify-between items-center mb-1">
                  <span className="text-xs font-medium text-gray-600">{label}</span>
                  <span className="text-xs font-bold text-gray-700">{value}</span>
                </div>
                <div className="w-full h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${color} transition-all duration-300`}
                    style={{ width: `${percentage}%` }}
                  />
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

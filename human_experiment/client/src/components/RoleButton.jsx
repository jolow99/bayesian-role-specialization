import React from "react";

export function RoleButton({ role, selected, onClick, disabled, locked }) {
  const roleConfig = {
    FIGHTER: {
      label: "Fighter",
      icon: "⚔️",
      description: "Attacks enemies (1-ε), random action otherwise",
      color: "red"
    },
    TANK: {
      label: "Tank",
      icon: "🛡️",
      description: "Defends when enemy attacks (1-ε), otherwise attacks",
      color: "blue"
    },
    HEALER: {
      label: "Healer",
      icon: "💚",
      description: "Heals when team health ≤50% (1-ε), otherwise attacks",
      color: "green"
    }
  };

  const config = roleConfig[role];

  const baseClasses = "flex flex-col items-center justify-center p-6 rounded-lg border-2 transition-all duration-200";

  const colorClasses = {
    red: locked
      ? "bg-red-100 border-red-500 shadow-lg ring-4 ring-red-300"
      : selected
        ? "bg-red-100 border-red-500 shadow-lg"
        : "bg-white border-red-300 hover:border-red-400 hover:bg-red-50 cursor-pointer hover:scale-105",
    blue: locked
      ? "bg-blue-100 border-blue-500 shadow-lg ring-4 ring-blue-300"
      : selected
        ? "bg-blue-100 border-blue-500 shadow-lg"
        : "bg-white border-blue-300 hover:border-blue-400 hover:bg-blue-50 cursor-pointer hover:scale-105",
    green: locked
      ? "bg-green-100 border-green-500 shadow-lg ring-4 ring-green-300"
      : selected
        ? "bg-green-100 border-green-500 shadow-lg"
        : "bg-white border-green-300 hover:border-green-400 hover:bg-green-50 cursor-pointer hover:scale-105"
  };

  const disabledClasses = "opacity-50 cursor-not-allowed hover:scale-100";

  return (
    <button
      onClick={onClick}
      disabled={disabled || locked}
      className={`${baseClasses} ${colorClasses[config.color]} ${(disabled && !locked) ? disabledClasses : ""}`}
    >
      <div className="text-5xl mb-2">{config.icon}</div>
      <div className="text-xl font-bold text-gray-800 mb-1">{config.label}</div>
      <div className="text-xs text-gray-600 text-center">{config.description}</div>
      {locked && (
        <div className="mt-2 text-xs font-bold text-gray-700 bg-white px-2 py-1 rounded">
          ACTIVE
        </div>
      )}
    </button>
  );
}

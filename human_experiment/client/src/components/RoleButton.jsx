import React from "react";

export function RoleButton({ role, selected, onClick, disabled, locked }) {
  const roleConfig = {
    FIGHTER: {
      label: "Fighter",
      icon: "🤺",
      description: "Attacks most of the time.",
      color: "red"
    },
    TANK: {
      label: "Tank",
      icon: "💂",
      description: "Defends when the enemy is attacking, most of the time. Otherwise, acts like a fighter.",
      color: "blue"
    },
    HEALER: {
      label: "Healer",
      icon: "🧙",
      description: "Heals if the team's health is less than 50%, most of the time. Otherwise, acts like a fighter.",
      color: "green"
    }
  };

  const config = roleConfig[role];

  const baseClasses = "flex flex-col items-center justify-center p-4 rounded-lg border-2 transition-all duration-200";

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
      <div className="text-4xl mb-1">{config.icon}</div>
      <div className="text-lg font-bold text-gray-800 mb-1">{config.label}</div>
      <div className="text-xs text-gray-600 text-center">{config.description}</div>
      {locked && (
        <div className="mt-1 text-xs font-bold text-gray-700 bg-white px-2 py-1 rounded">
          ACTIVE
        </div>
      )}
    </button>
  );
}

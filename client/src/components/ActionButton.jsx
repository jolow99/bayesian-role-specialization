import React from "react";

export function ActionButton({ action, selected, onClick, disabled }) {
  const actionConfig = {
    ATTACK: {
      label: "Attack",
      icon: "⚔️",
      description: "Deal damage to enemy",
      color: "red"
    },
    DEFEND: {
      label: "Defend",
      icon: "🛡️",
      description: "Protect team from enemy attacks",
      color: "blue"
    },
    HEAL: {
      label: "Heal",
      icon: "💚",
      description: "Restore team health",
      color: "green"
    }
  };

  const config = actionConfig[action];

  const baseClasses = "flex flex-col items-center justify-center p-6 rounded-lg border-2 transition-all duration-200 cursor-pointer hover:scale-105";

  const colorClasses = {
    red: selected
      ? "bg-red-100 border-red-500 shadow-lg"
      : "bg-white border-red-300 hover:border-red-400 hover:bg-red-50",
    blue: selected
      ? "bg-blue-100 border-blue-500 shadow-lg"
      : "bg-white border-blue-300 hover:border-blue-400 hover:bg-blue-50",
    green: selected
      ? "bg-green-100 border-green-500 shadow-lg"
      : "bg-white border-green-300 hover:border-green-400 hover:bg-green-50"
  };

  const disabledClasses = "opacity-50 cursor-not-allowed hover:scale-100";

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`${baseClasses} ${colorClasses[config.color]} ${disabled ? disabledClasses : ""}`}
    >
      <div className="text-5xl mb-2">{config.icon}</div>
      <div className="text-xl font-bold text-gray-800 mb-1">{config.label}</div>
      <div className="text-xs text-gray-600 text-center">{config.description}</div>
    </button>
  );
}

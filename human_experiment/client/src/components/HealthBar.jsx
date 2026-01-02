import React from "react";

export function HealthBar({ label, current, max, color = "green", previousHealth = null, damageAmount = 0, healAmount = 0, showChange = false }) {
  const percentage = Math.max(0, Math.min(100, (current / max) * 100));
  const previousPercentage = previousHealth !== null ? Math.max(0, Math.min(100, (previousHealth / max) * 100)) : percentage;

  const colorClasses = {
    green: "bg-green-500",
    red: "bg-red-500",
    blue: "bg-blue-500"
  };

  const bgColor = colorClasses[color] || colorClasses.green;

  // Calculate health change text - always show if damage or heal occurred
  let changeText = "";
  if (showChange && (damageAmount > 0 || healAmount > 0)) {
    if (damageAmount > 0 && healAmount > 0) {
      // Both damage and healing - always show both
      changeText = ` (-${damageAmount} +${healAmount})`;
    } else if (damageAmount > 0) {
      changeText = ` (-${damageAmount})`;
    } else if (healAmount > 0) {
      changeText = ` (+${healAmount})`;
    }
  }

  return (
    <div className="w-full">
      {label && (
        <div className="flex justify-between items-center mb-1">
          <span className="text-sm font-semibold text-gray-700">{label}</span>
        </div>
      )}
      <div className="w-full h-8 bg-gray-200 rounded-full overflow-hidden border-2 border-gray-400 relative">
        {/* Previous health bar (transparent overlay) - shows where health was before */}
        {showChange && previousHealth !== null && previousPercentage !== percentage && (
          <div
            className={`absolute h-full ${bgColor} opacity-30`}
            style={{ width: `${previousPercentage}%` }}
          />
        )}
        {/* Current health bar */}
        <div
          className={`absolute h-full ${bgColor} transition-all duration-500 ease-out`}
          style={{ width: `${percentage}%` }}
        />
        {/* Health text centered on the bar */}
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-sm font-bold text-gray-800 drop-shadow-[0_1px_1px_rgba(255,255,255,0.8)]">
            {current % 1 === 0 ? current : current.toFixed(1)}/{max}
            {changeText && (
              <span className={damageAmount > 0 && healAmount === 0 ? "text-red-600" : healAmount > 0 && damageAmount === 0 ? "text-green-600" : "text-purple-700"}>
                {changeText}
              </span>
            )}
          </span>
        </div>
      </div>
    </div>
  );
}

import React from "react";

export function HealthBar({ label, current, max, color = "green" }) {
  const percentage = Math.max(0, Math.min(100, (current / max) * 100));

  const colorClasses = {
    green: "bg-green-500",
    red: "bg-red-500",
    blue: "bg-blue-500"
  };

  const bgColor = colorClasses[color] || colorClasses.green;

  return (
    <div className="w-full">
      <div className="flex justify-between items-center mb-1">
        <span className="text-sm font-semibold text-gray-700">{label}</span>
        <span className="text-sm font-medium text-gray-600">
          {current % 1 === 0 ? current : current.toFixed(1)}/{max}
        </span>
      </div>
      <div className="w-full h-6 bg-gray-200 rounded-full overflow-hidden border border-gray-300">
        <div
          className={`h-full ${bgColor} transition-all duration-500 ease-out flex items-center justify-center`}
          style={{ width: `${percentage}%` }}
        >
          {percentage > 15 && (
            <span className="text-xs font-bold text-white">
              {Math.round(percentage)}%
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

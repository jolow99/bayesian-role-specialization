import React, { useState } from "react";

export function StatsInfo() {
  const [showTooltip, setShowTooltip] = useState(false);

  return (
    <div className="relative inline-block">
      <button
        className="text-white bg-gray-700 hover:bg-gray-600 font-semibold text-xs px-3 py-1 rounded-full flex items-center gap-1 transition-colors"
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
        data-tutorial-id="stats-info"
      >
        ℹ️ How do stats work?
      </button>

      {showTooltip && (
        <div className="absolute z-50 bg-white border-2 border-gray-300 rounded-lg shadow-xl p-4 w-80 bottom-full mb-2 right-0" data-tutorial-id="stats-info-tooltip">
          <div className="text-xs space-y-2">
            <div className="font-bold text-gray-800 mb-3">How Stats Work</div>

            <div className="space-y-2">
              <div>
                <div className="font-semibold text-red-600 mb-1">⚔️ Attack</div>
                <div className="text-gray-700">The amount of damage taken by a boss is the sum of all STR stats of players who attack</div>
              </div>

              <div>
                <div className="font-semibold text-blue-600 mb-1">🛡️ Block</div>
                <div className="text-gray-700">The amount of damage blocked by the team is the highest DEF stat amongst players who defend</div>
              </div>

              <div>
                <div className="font-semibold text-green-600 mb-1">💚 Heal</div>
                <div className="text-gray-700">The amount of health healed by the team is the sum of all SUP stats of players who heal</div>
              </div>
            </div>

            <div className="text-gray-600 italic mt-3 pt-3 border-t border-gray-200">
              All roles can perform any action, but they follow different strategies.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

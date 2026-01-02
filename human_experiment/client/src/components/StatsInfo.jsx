import React, { useState } from "react";

export function StatsInfo() {
  const [showTooltip, setShowTooltip] = useState(false);

  return (
    <div className="relative inline-block">
      <button
        className="text-white bg-gray-700 hover:bg-gray-600 font-semibold text-xs px-3 py-1 rounded-full flex items-center gap-1 transition-colors"
        onMouseEnter={() => setShowTooltip(true)}
        onMouseLeave={() => setShowTooltip(false)}
      >
        ℹ️ How do stats work?
      </button>

      {showTooltip && (
        <div className="absolute z-10 bg-white border-2 border-gray-300 rounded-lg shadow-xl p-4 w-80 -top-2 left-full ml-2">
          <div className="text-xs space-y-2">
            <div className="font-bold text-gray-800 mb-3">How Stats Work</div>

            <div className="space-y-2">
              <div>
                <div className="font-semibold text-red-600 mb-1">⚔️ Attack</div>
                <div className="text-gray-700">Each attacker deals damage equal to their STR. Total damage = sum of all attackers' STR.</div>
              </div>

              <div>
                <div className="font-semibold text-blue-600 mb-1">🛡️ Defend</div>
                <div className="text-gray-700">Incoming damage is reduced by the highest DEF among all defenders.</div>
              </div>

              <div>
                <div className="font-semibold text-green-600 mb-1">💚 Heal</div>
                <div className="text-gray-700">Each healer restores health equal to their SUP. Total healing = sum of all healers' SUP (up to max team health).</div>
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

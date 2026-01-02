import React from "react";
import { RoleButton } from "./RoleButton";
import { StatsInfo } from "./StatsInfo";

const ROLES = { FIGHTER: 0, TANK: 1, HEALER: 2 };

export const ActionMenu = React.memo(function ActionMenu({
  selectedRole,
  onRoleSelect,
  onSubmit,
  isRoleCommitted,
  currentRole,
  roundsRemaining,
  submitted,
  roleOrder = [ROLES.FIGHTER, ROLES.TANK, ROLES.HEALER] // Default order if not provided
}) {
  const ROLE_NAMES = ["FIGHTER", "TANK", "HEALER"];
  const orderedRoleNames = roleOrder.map(roleId => ROLE_NAMES[roleId]);
  return (
    <div>
      <div className="bg-gray-800 text-white rounded-t-lg px-4 py-2 text-sm font-bold flex items-center justify-between">
        <span>
          {isRoleCommitted
            ? "Your role is locked for this round"
            : "What role will you play?"}
        </span>
        <StatsInfo />
      </div>
      <div className="bg-gray-100 rounded-b-lg border-2 border-gray-800 border-t-0 p-4">
        <div className="grid grid-cols-3 gap-3 mb-3">
          {orderedRoleNames.map((roleName) => {
            const roleValue = ROLES[roleName];
            return (
              <RoleButton
                key={roleName}
                role={roleName}
                selected={selectedRole === roleValue}
                onClick={() => onRoleSelect(roleValue)}
                disabled={submitted}
                locked={isRoleCommitted && currentRole === roleValue}
              />
            );
          })}
        </div>

        <button
          onClick={onSubmit}
          disabled={!isRoleCommitted && selectedRole === null}
          className={`w-full py-3 px-4 rounded-lg font-bold text-white transition-all ${
            (!isRoleCommitted && selectedRole === null)
              ? "bg-gray-400 cursor-not-allowed"
              : isRoleCommitted
                ? "bg-green-600 hover:bg-green-700 shadow-lg"
                : "bg-blue-600 hover:bg-blue-700 shadow-lg"
          }`}
        >
          {isRoleCommitted
            ? `▶ READY (${roundsRemaining} rounds left)`
            : selectedRole === null
              ? "Select a role"
              : "✓ CONFIRM ROLE (2 rounds)"}
        </button>
      </div>
    </div>
  );
});

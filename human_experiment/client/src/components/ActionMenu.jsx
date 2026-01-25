import React from "react";
import { RoleButton } from "./RoleButton";
import { StatsInfo } from "./StatsInfo";
import { ROLES, ROLE_NAMES, ROLE_CONFIG } from "../constants";

// Small role selector for inference
const InferenceRoleSelector = React.memo(function InferenceRoleSelector({
  playerId,
  selectedRole,
  onSelect,
  disabled
}) {
  const roles = [ROLES.FIGHTER, ROLES.TANK, ROLES.MEDIC];

  return (
    <div className="flex flex-col items-center">
      <span className="text-xs font-semibold text-gray-600 mb-1">P{playerId + 1}'s Role</span>
      <div className="flex gap-1">
        {roles.map((roleValue) => {
          const roleName = ROLE_NAMES[roleValue];
          const config = ROLE_CONFIG[roleName];
          const isSelected = selectedRole === roleValue;

          return (
            <button
              key={roleValue}
              onClick={() => onSelect(playerId, roleValue)}
              disabled={disabled}
              title={config.label}
              className={`w-14 h-14 rounded-lg border-2 transition-all flex items-center justify-center text-2xl
                ${isSelected
                  ? 'border-blue-500 bg-blue-100 shadow-md'
                  : 'border-gray-300 bg-white hover:border-gray-400 hover:bg-gray-50'}
                ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}
              `}
            >
              {config.icon}
            </button>
          );
        })}
      </div>
    </div>
  );
});

export const ActionMenu = React.memo(function ActionMenu({
  selectedRole,
  onRoleSelect,
  onSubmit,
  isRoleCommitted,
  currentRole,
  roundsRemaining,
  submitted,
  roleOrder = [ROLES.FIGHTER, ROLES.TANK, ROLES.MEDIC], // Default order if not provided
  otherPlayersStatus = [],
  // New props for inference reporting
  inferredRoles = {}, // { playerId: roleValue } - what user thinks each player's role is
  onInferredRoleChange = () => {}, // (playerId, roleValue) => void
  showInference = true, // Whether to show inference UI (can be disabled for tutorials)
  isFirstStage = false // Whether this is the first stage (no past turns to infer from)
}) {
  const orderedRoleNames = roleOrder.map(roleId => ROLE_NAMES[roleId]);

  // Check if all inferences are complete
  const otherPlayerIds = otherPlayersStatus.map(p => p.playerId);
  const allInferencesComplete = (showInference && !isFirstStage)
    ? otherPlayerIds.every(pid => inferredRoles[pid] !== undefined && inferredRoles[pid] !== null)
    : true;

  // Can submit only if role selected AND (inferences complete OR inference not shown)
  const canSubmit = isRoleCommitted || (selectedRole !== null && allInferencesComplete);

  return (
    <div data-tutorial-id="action-menu" className="space-y-3">
      {/* Inference card - what do you think other players' roles are? */}
      {showInference && !isFirstStage && otherPlayersStatus.length > 0 && !isRoleCommitted && (
        <div data-tutorial-id="inference-section">
          <div className="bg-amber-600 text-white rounded-t-lg px-4 py-2 text-sm font-bold">
            What roles do you think your teammates played in the past stage?
          </div>
          <div className="bg-amber-50 rounded-b-lg border-2 border-amber-600 border-t-0 p-4">
            <div className="flex justify-center gap-8">
              {otherPlayersStatus.map((p) => (
                <InferenceRoleSelector
                  key={p.odId}
                  playerId={p.playerId}
                  selectedRole={inferredRoles[p.playerId]}
                  onSelect={onInferredRoleChange}
                  disabled={submitted}
                />
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Role selection card */}
      <div>
        <div className="bg-gray-800 text-white rounded-t-lg px-4 py-2 text-sm font-bold flex items-center justify-between">
          <span>
            {isRoleCommitted
              ? "Your role is locked for this round"
              : "What role will you play?"}
          </span>
          <div className="flex items-center gap-4">
            {/* Show other players' submission status */}
            {otherPlayersStatus.length > 0 && (
              <div className="flex items-center gap-3 text-xs">
                {otherPlayersStatus.map((p) => (
                  <div key={p.odId} className="flex items-center gap-1">
                    <span className="text-gray-300">P{(p.playerId ?? 0) + 1}:</span>
                    {p.submitted ? (
                      <span className="text-green-400 font-semibold">✓</span>
                    ) : (
                      <span className="text-orange-400 animate-pulse">...</span>
                    )}
                  </div>
                ))}
              </div>
            )}
            <StatsInfo />
          </div>
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
            disabled={!canSubmit}
            className={`w-full py-3 px-4 rounded-lg font-bold text-white transition-all ${
              !canSubmit
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
                : !allInferencesComplete
                  ? "Select teammate roles to continue"
                  : "✓ CONFIRM ROLE (2 rounds)"}
          </button>
        </div>
      </div>
    </div>
  );
});

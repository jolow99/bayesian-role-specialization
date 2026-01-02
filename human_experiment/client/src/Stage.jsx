import React from "react";
import ActionSelection from "./stages/ActionSelection";

// Stage component just returns ActionSelection - no subscriptions needed here
// ActionSelection handles all data subscriptions internally to minimize re-renders
export function Stage() {
  return <ActionSelection />;
}

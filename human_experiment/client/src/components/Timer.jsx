import { useStageTimer } from "@empirica/core/player/classic/react";
import { usePlayer } from "@empirica/core/player/classic/react";
import React from "react";
import { STAGE_TIMER_SECONDS, BUFFER_TOTAL_SECONDS } from "../constants";

export function Timer() {
  const timer = useStageTimer();
  const player = usePlayer();

  let totalRemaining;
  if (timer?.remaining || timer?.remaining === 0) {
    totalRemaining = Math.round(timer?.remaining / 1000);
  }

  const submitted = player?.stage?.get("submit");
  const bufferTimeRemaining = player?.get("bufferTimeRemaining") ?? BUFFER_TOTAL_SECONDS;

  // Stage timer: first 90s of the Empirica stage duration
  // Buffer zone starts at BUFFER_TOTAL_SECONDS remaining on the Empirica timer
  const stageSeconds = totalRemaining !== undefined
    ? Math.max(0, totalRemaining - BUFFER_TOTAL_SECONDS)
    : null;

  // Buffer display: how much buffer the player actually has left
  // When stage timer is running (stageSeconds > 0), buffer is not active
  // When stage timer expired, buffer counts down
  const bufferSeconds = totalRemaining !== undefined && stageSeconds === 0
    ? Math.min(totalRemaining, bufferTimeRemaining)
    : null;

  const stageExpired = stageSeconds === 0 && totalRemaining !== null;
  const bufferLow = bufferSeconds !== null && bufferSeconds < 60;
  const bufferCritical = bufferSeconds !== null && bufferSeconds < 30;

  // If submitted, just show stage timer (no urgency needed)
  if (submitted) {
    return (
      <div className="flex flex-col items-center">
        <h1 className="tabular-nums text-3xl text-gray-400 font-semibold">
          {humanTimer(stageSeconds)}
        </h1>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center gap-0.5">
      {/* Stage timer */}
      <h1 className={`tabular-nums text-3xl font-semibold ${
        stageExpired ? "text-red-500" : "text-gray-500"
      }`}>
        {humanTimer(stageSeconds)}
      </h1>

      {/* Buffer timer - only shown when stage timer expired */}
      {stageExpired && !submitted && (
        <div className={`flex items-center gap-1 text-xs font-medium ${
          bufferCritical ? "text-red-600 animate-pulse" :
          bufferLow ? "text-amber-600" :
          "text-amber-500"
        }`}>
          <span>Buffer:</span>
          <span className="tabular-nums font-semibold">
            {humanTimer(bufferSeconds)}
          </span>
        </div>
      )}

      {/* Buffer remaining indicator (always visible, small) */}
      {!stageExpired && bufferTimeRemaining < BUFFER_TOTAL_SECONDS && (
        <div className="text-xs text-gray-400 tabular-nums">
          Buffer: {humanTimer(Math.round(bufferTimeRemaining))}
        </div>
      )}
    </div>
  );
}

function humanTimer(seconds) {
  if (seconds === null || seconds === undefined) {
    return "--:--";
  }

  let out = "";
  const s = seconds % 60;
  out += s < 10 ? "0" + s : s;

  const min = (seconds - s) / 60;
  if (min === 0) {
    return `00:${out}`;
  }

  const m = min % 60;
  out = `${m < 10 ? "0" + m : m}:${out}`;

  const h = (min - m) / 60;
  if (h === 0) {
    return out;
  }

  return `${h}:${out}`;
}

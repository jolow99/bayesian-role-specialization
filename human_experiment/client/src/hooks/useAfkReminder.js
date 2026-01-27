import { useEffect, useRef, useCallback } from "react";

// Configuration - tweak these values as needed
const AFK_CONFIG = {
  // Time thresholds (in milliseconds)
  INACTIVE_THRESHOLD: 20000,        // 20s of no activity before first reminder
  TAB_HIDDEN_THRESHOLD: 8000,       // 8s of tab hidden before first reminder

  // When teammates are waiting, be more aggressive
  TEAMMATES_WAITING_INACTIVE: 12000, // 12s if 1+ teammates waiting
  TEAMMATES_WAITING_TAB_HIDDEN: 4000, // 4s if 1+ teammates waiting
  ALL_TEAMMATES_WAITING_INACTIVE: 8000, // 8s if ALL teammates waiting
  ALL_TEAMMATES_WAITING_TAB_HIDDEN: 3000, // 3s if ALL teammates waiting

  REMINDER_INTERVAL: 15000,          // 15s between subsequent reminders (base)
  REMINDER_INTERVAL_URGENT: 10000,   // 10s if teammates waiting
  ESCALATION_INTERVAL: 2,            // Escalate annoyance every N reminders

  // Audio settings
  BEEP_FREQUENCY: 800,              // Hz
  BEEP_DURATION: 150,               // ms
  BEEP_COUNT: 3,                    // Number of beeps
  BEEP_GAP: 200,                    // ms between beeps
};

// Different messages based on urgency
const MESSAGES = {
  GENTLE: "Please make your selection.",
  TEAMMATE_WAITING: "Your teammate is waiting for you.",
  TEAMMATES_WAITING: "Your teammates are waiting for you.",
  URGENT: "Your teammates are waiting! Please choose quickly.",
};

// Create beep sound using Web Audio API
function playBeeps(count = 3, escalationLevel = 0) {
  try {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return;

    const audioCtx = new AudioContext();
    const baseFrequency = AFK_CONFIG.BEEP_FREQUENCY + (escalationLevel * 100); // Higher pitch = more urgent
    const volume = Math.min(0.3 + (escalationLevel * 0.1), 0.7); // Louder over time

    for (let i = 0; i < count; i++) {
      const startTime = audioCtx.currentTime + (i * (AFK_CONFIG.BEEP_DURATION + AFK_CONFIG.BEEP_GAP)) / 1000;

      const oscillator = audioCtx.createOscillator();
      const gainNode = audioCtx.createGain();

      oscillator.connect(gainNode);
      gainNode.connect(audioCtx.destination);

      oscillator.frequency.value = baseFrequency;
      oscillator.type = "sine";

      gainNode.gain.setValueAtTime(volume, startTime);
      gainNode.gain.exponentialRampToValueAtTime(0.01, startTime + AFK_CONFIG.BEEP_DURATION / 1000);

      oscillator.start(startTime);
      oscillator.stop(startTime + AFK_CONFIG.BEEP_DURATION / 1000);
    }

    // Close audio context after sounds finish
    setTimeout(() => {
      audioCtx.close();
    }, count * (AFK_CONFIG.BEEP_DURATION + AFK_CONFIG.BEEP_GAP) + 100);

  } catch (e) {
    console.warn("Could not play beep sound:", e);
  }
}

// Speak a message using Web Speech API
function speakMessage(message, escalationLevel = 0) {
  try {
    if (!window.speechSynthesis) return;

    // Cancel any ongoing speech
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(message);
    utterance.rate = 1.0 + (escalationLevel * 0.1); // Faster = more urgent
    utterance.pitch = 1.0;
    utterance.volume = Math.min(0.8 + (escalationLevel * 0.05), 1.0);

    window.speechSynthesis.speak(utterance);
  } catch (e) {
    console.warn("Could not speak message:", e);
  }
}

// Play full reminder (beeps + speech)
function playReminder(escalationLevel = 0, message = MESSAGES.GENTLE) {
  const beepCount = AFK_CONFIG.BEEP_COUNT + Math.min(escalationLevel, 3); // More beeps over time
  playBeeps(beepCount, escalationLevel);

  // Speak after beeps finish
  const beepDuration = beepCount * (AFK_CONFIG.BEEP_DURATION + AFK_CONFIG.BEEP_GAP);
  setTimeout(() => {
    speakMessage(message, escalationLevel);
  }, beepDuration);
}

/**
 * Hook to detect AFK behavior and play reminders when the player needs to act.
 *
 * Only triggers when:
 * - Player is in role selection phase (needsInput = true)
 * - Player hasn't submitted yet
 *
 * Escalates based on:
 * - How many teammates are already waiting (submitted)
 * - How long the player has been inactive
 *
 * @param {Object} options
 * @param {boolean} options.needsInput - True when the player needs to make a selection
 * @param {boolean} options.hasSubmitted - Whether this player has already submitted
 * @param {Array} options.otherPlayersStatus - Array of {submitted: boolean} for other players
 * @param {Function} options.onAfkDetected - Optional callback when AFK is detected
 * @param {Function} options.onActivityResumed - Optional callback when user becomes active again
 */
export function useAfkReminder({
  needsInput = false,
  hasSubmitted = false,
  otherPlayersStatus = [],
  onAfkDetected = null,
  onActivityResumed = null,
} = {}) {
  const lastActivityRef = useRef(Date.now());
  const tabHiddenSinceRef = useRef(null);
  const reminderCountRef = useRef(0);
  const reminderIntervalRef = useRef(null);
  const isAfkRef = useRef(false);
  const checkTimeoutRef = useRef(null);

  // Calculate how many teammates are waiting
  const getTeammatesWaiting = useCallback(() => {
    return otherPlayersStatus.filter(p => p.submitted).length;
  }, [otherPlayersStatus]);

  const getTotalTeammates = useCallback(() => {
    return otherPlayersStatus.length;
  }, [otherPlayersStatus]);

  // Get appropriate thresholds based on teammates waiting
  const getThresholds = useCallback(() => {
    const waiting = getTeammatesWaiting();
    const total = getTotalTeammates();

    if (total > 0 && waiting === total) {
      // ALL teammates waiting - most urgent
      return {
        inactive: AFK_CONFIG.ALL_TEAMMATES_WAITING_INACTIVE,
        tabHidden: AFK_CONFIG.ALL_TEAMMATES_WAITING_TAB_HIDDEN,
        interval: AFK_CONFIG.REMINDER_INTERVAL_URGENT,
      };
    } else if (waiting > 0) {
      // Some teammates waiting - urgent
      return {
        inactive: AFK_CONFIG.TEAMMATES_WAITING_INACTIVE,
        tabHidden: AFK_CONFIG.TEAMMATES_WAITING_TAB_HIDDEN,
        interval: AFK_CONFIG.REMINDER_INTERVAL_URGENT,
      };
    } else {
      // No teammates waiting yet - gentle
      return {
        inactive: AFK_CONFIG.INACTIVE_THRESHOLD,
        tabHidden: AFK_CONFIG.TAB_HIDDEN_THRESHOLD,
        interval: AFK_CONFIG.REMINDER_INTERVAL,
      };
    }
  }, [getTeammatesWaiting, getTotalTeammates]);

  // Get appropriate message based on state
  const getMessage = useCallback((escalationLevel) => {
    const waiting = getTeammatesWaiting();
    const total = getTotalTeammates();

    if (escalationLevel >= 2 && waiting > 0) {
      return MESSAGES.URGENT;
    } else if (waiting === total && total > 0) {
      return total === 1 ? MESSAGES.TEAMMATE_WAITING : MESSAGES.TEAMMATES_WAITING;
    } else if (waiting > 0) {
      return waiting === 1 ? MESSAGES.TEAMMATE_WAITING : MESSAGES.TEAMMATES_WAITING;
    } else {
      return MESSAGES.GENTLE;
    }
  }, [getTeammatesWaiting, getTotalTeammates]);

  // Reset activity timestamp
  const resetActivity = useCallback(() => {
    lastActivityRef.current = Date.now();

    // If we were AFK, mark as active again
    if (isAfkRef.current) {
      isAfkRef.current = false;
      reminderCountRef.current = 0;

      if (onActivityResumed) {
        onActivityResumed();
      }
    }
  }, [onActivityResumed]);

  // Check if user is AFK and play reminder if needed
  const checkAfk = useCallback(() => {
    // Only check if player needs to provide input and hasn't submitted
    if (!needsInput || hasSubmitted) {
      return;
    }

    const now = Date.now();
    const isTabHidden = document.visibilityState === "hidden";
    const timeSinceActivity = now - lastActivityRef.current;
    const timeSinceTabHidden = tabHiddenSinceRef.current
      ? now - tabHiddenSinceRef.current
      : 0;

    const thresholds = getThresholds();

    // Determine if we should trigger a reminder
    const shouldRemind =
      (isTabHidden && timeSinceTabHidden >= thresholds.tabHidden) ||
      (!isTabHidden && timeSinceActivity >= thresholds.inactive);

    if (shouldRemind) {
      if (!isAfkRef.current) {
        isAfkRef.current = true;
        if (onAfkDetected) {
          onAfkDetected();
        }
      }

      const escalationLevel = Math.floor(reminderCountRef.current / AFK_CONFIG.ESCALATION_INTERVAL);
      const message = getMessage(escalationLevel);
      playReminder(escalationLevel, message);
      reminderCountRef.current++;
    }
  }, [needsInput, hasSubmitted, getThresholds, getMessage, onAfkDetected]);

  // Handle visibility change
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.visibilityState === "hidden") {
        tabHiddenSinceRef.current = Date.now();
      } else {
        tabHiddenSinceRef.current = null;
        resetActivity();
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [resetActivity]);

  // Handle user activity events
  useEffect(() => {
    if (!needsInput || hasSubmitted) return;

    const events = ["mousedown", "mousemove", "keydown", "touchstart", "scroll"];

    // Throttle activity updates to avoid performance issues
    let throttleTimeout = null;
    const throttledReset = () => {
      if (throttleTimeout) return;
      throttleTimeout = setTimeout(() => {
        throttleTimeout = null;
        resetActivity();
      }, 100);
    };

    events.forEach(event => {
      document.addEventListener(event, throttledReset, { passive: true });
    });

    return () => {
      events.forEach(event => {
        document.removeEventListener(event, throttledReset);
      });
      if (throttleTimeout) {
        clearTimeout(throttleTimeout);
      }
    };
  }, [needsInput, hasSubmitted, resetActivity]);

  // Set up AFK checking - re-run when teammates submit (thresholds change)
  useEffect(() => {
    // Clear existing timers
    if (checkTimeoutRef.current) {
      clearTimeout(checkTimeoutRef.current);
      checkTimeoutRef.current = null;
    }
    if (reminderIntervalRef.current) {
      clearInterval(reminderIntervalRef.current);
      reminderIntervalRef.current = null;
    }

    if (!needsInput || hasSubmitted) {
      return;
    }

    // Reset state when we start monitoring
    lastActivityRef.current = Date.now();
    reminderCountRef.current = 0;
    isAfkRef.current = false;

    const thresholds = getThresholds();

    // Check after the threshold for current urgency level
    const initialCheckDelay = Math.min(thresholds.inactive, thresholds.tabHidden);

    checkTimeoutRef.current = setTimeout(() => {
      checkAfk();

      // Then check at regular intervals based on urgency
      reminderIntervalRef.current = setInterval(checkAfk, thresholds.interval);
    }, initialCheckDelay);

    return () => {
      if (checkTimeoutRef.current) {
        clearTimeout(checkTimeoutRef.current);
        checkTimeoutRef.current = null;
      }
      if (reminderIntervalRef.current) {
        clearInterval(reminderIntervalRef.current);
        reminderIntervalRef.current = null;
      }
    };
  }, [needsInput, hasSubmitted, checkAfk, getThresholds]);

  // Re-check when teammates submit (urgency may have changed)
  useEffect(() => {
    if (needsInput && !hasSubmitted && isAfkRef.current) {
      // Player is already marked AFK, and teammates status changed
      // Play an immediate reminder with updated urgency
      const escalationLevel = Math.floor(reminderCountRef.current / AFK_CONFIG.ESCALATION_INTERVAL);
      const message = getMessage(escalationLevel);
      playReminder(escalationLevel, message);
      reminderCountRef.current++;
    }
  }, [otherPlayersStatus, needsInput, hasSubmitted, getMessage]);

  // Return methods to manually trigger/reset if needed
  return {
    resetActivity,
    playReminderNow: () => {
      const escalationLevel = Math.floor(reminderCountRef.current / AFK_CONFIG.ESCALATION_INTERVAL);
      playReminder(escalationLevel, getMessage(escalationLevel));
    },
  };
}

export default useAfkReminder;

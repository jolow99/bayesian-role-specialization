import React, { useState, useEffect } from "react";

export function PlayerCreate({ onPlayerID, connecting }) {
  const [playerID, setPlayerID] = useState("");
  const [autoSubmitted, setAutoSubmitted] = useState(false);

  useEffect(() => {
    // Extract Prolific params from URL
    const params = new URLSearchParams(window.location.search);
    const prolificPid = params.get("PROLIFIC_PID");
    const studyId = params.get("STUDY_ID");
    const sessionId = params.get("SESSION_ID");

    // Store all Prolific params in sessionStorage for later use
    if (prolificPid) {
      sessionStorage.setItem("PROLIFIC_PID", prolificPid);
    }
    if (studyId) {
      sessionStorage.setItem("STUDY_ID", studyId);
    }
    if (sessionId) {
      sessionStorage.setItem("SESSION_ID", sessionId);
    }

    if (prolificPid && prolificPid.trim() !== "" && !autoSubmitted) {
      // Auto-submit the Prolific PID
      setAutoSubmitted(true);
      onPlayerID(prolificPid.trim());
    }
  }, [onPlayerID, autoSubmitted]);

  // If we have a PROLIFIC_PID, show a loading state while auto-submitting
  const params = new URLSearchParams(window.location.search);
  const hasProlificPid = params.get("PROLIFIC_PID");

  if (hasProlificPid) {
    return (
      <div className="min-h-screen bg-gray-100 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
        <div className="sm:mx-auto sm:w-full sm:max-w-md">
          <h2 className="text-center text-2xl font-bold text-gray-900">
            Connecting...
          </h2>
          <p className="mt-2 text-center text-sm text-gray-600">
            Please wait while we set up your session.
          </p>
        </div>
      </div>
    );
  }

  // Fallback form for testing without Prolific URL params
  const handleSubmit = (evt) => {
    evt.preventDefault();
    if (!playerID || playerID.trim() === "") {
      return;
    }
    onPlayerID(playerID.trim());
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
      <div className="sm:mx-auto sm:w-full sm:max-w-md">
        <h2 className="text-center text-2xl font-bold text-gray-900">
          Enter Your Prolific ID
        </h2>
        <p className="mt-2 text-center text-sm text-gray-600">
          Please enter your prolific ID to continue.
        </p>
      </div>

      <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
        <div className="bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10">
          <form onSubmit={handleSubmit}>
            <fieldset disabled={connecting}>
              <div>
                <label
                  htmlFor="playerID"
                  className="block text-sm font-medium text-gray-700"
                >
                  Prolific ID
                </label>
                <div className="mt-1">
                  <input
                    id="playerID"
                    type="text"
                    autoComplete="off"
                    value={playerID}
                    onChange={(e) => setPlayerID(e.target.value)}
                    className="appearance-none block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm"
                    placeholder="Enter your ID"
                  />
                </div>
              </div>

              <div className="mt-6">
                <button
                  type="submit"
                  className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {connecting ? "Connecting..." : "Continue"}
                </button>
              </div>
            </fieldset>
          </form>
        </div>
      </div>
    </div>
  );
}

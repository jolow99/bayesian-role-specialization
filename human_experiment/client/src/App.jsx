import { EmpiricaClassic } from "@empirica/core/player/classic";
import { EmpiricaContext } from "@empirica/core/player/classic/react";
import { EmpiricaMenu, EmpiricaParticipant } from "@empirica/core/player/react";
import React from "react";
import { Game } from "./Game";
import { Consent } from "./intro-exit/Consent";
import { ExitSurvey } from "./intro-exit/ExitSurvey";
import { Finished } from "./intro-exit/Finished";
import { Introduction } from "./intro-exit/Introduction";
import { Lobby } from "./intro-exit/Lobby";
import { PlayerCreate } from "./intro-exit/PlayerCreate";
import { Tutorial1 } from "./intro-exit/Tutorial1";
import { Tutorial2 } from "./intro-exit/Tutorial2";
import { ComprehensionSurvey } from "./intro-exit/ComprehensionSurvey";
import { SkeletonLoader } from "./components/SkeletonLoader";

export default function App() {
  const urlParams = new URLSearchParams(window.location.search);
  const playerKey = urlParams.get("participantKey") || "";

  const { protocol, host } = window.location;
  const url = `${protocol}//${host}/query`;

  function introSteps({ game, player }) {
    // Check if tutorial should be shown from treatment configuration
    const showTutorial = game?.get("treatment")?.showTutorial ?? true;

    if (showTutorial) {
      return [Introduction, Tutorial1, Tutorial2, ComprehensionSurvey];
    }
    return [Introduction];
  }

  function exitSteps({ game, player }) {
    // Skip survey if player never completed the game (lobby timeout or early exit)
    const gameComplete = player?.get("game_complete");
    if (!game || !gameComplete) {
      // Lobby timed out or game never completed - skip survey, go straight to Finished
      return [];
    }
    return [ExitSurvey];
  }

  return (
    <EmpiricaParticipant url={url} ns={playerKey} modeFunc={EmpiricaClassic}>
      <div className="h-screen relative">
        <EmpiricaMenu position="bottom-left" />
        <div className="h-full overflow-auto">
          <EmpiricaContext
            playerCreate={PlayerCreate}
            consent={Consent}
            lobby={Lobby}
            finished={Finished}
            introSteps={introSteps}
            exitSteps={exitSteps}
            loading={SkeletonLoader}
          >
            <Game />
          </EmpiricaContext>
        </div>
      </div>
    </EmpiricaParticipant>
  );
}

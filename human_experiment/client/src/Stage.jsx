import {
  usePlayer,
  usePlayers,
  useRound,
  useStage,
} from "@empirica/core/player/classic/react";
import { Loading } from "@empirica/core/player/react";
import React from "react";
import { ActionSelection } from "./stages/ActionSelection";
import { Reveal } from "./stages/Reveal";

export function Stage() {
  const player = usePlayer();
  const players = usePlayers();
  const round = useRound();
  const stage = useStage();

  const stageName = stage.get("name");

  console.log("Current stage:", stageName);

  switch (stageName) {
    case "Action Selection":
      return <ActionSelection />;
    case "Reveal":
      return <Reveal />;
    default:
      return <div>Unknown stage: {stageName}</div>;
  }
}

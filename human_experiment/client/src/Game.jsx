import { useGame } from "@empirica/core/player/classic/react";

import React from "react";
import { Profile } from "./Profile";
import { Stage } from "./Stage";

export function Game() {
  const game = useGame();

  return (
    <div className="h-full w-full flex">
      <div className="h-full w-full flex flex-col">
      <Profile />
        <div className="h-full flex items-center justify-center">
          <Stage key="stable-stage" />
        </div>
      </div>
    </div>
  );
}

import { useGame } from "@empirica/core/player/classic/react";

import React from "react";
import { Profile } from "./Profile";
import { Stage } from "./Stage";

export function Game() {
  const game = useGame();
  const containerRef = React.useRef(null);

  // Cache the current game state HTML continuously (after every render)
  // This ensures the cache is ALWAYS up-to-date before any unmount happens
  React.useEffect(() => {
    if (containerRef.current) {
      try {
        const html = containerRef.current.outerHTML; // Use outerHTML to include the container itself
        sessionStorage.setItem('gameStateCache', JSON.stringify({ html }));
        // Don't log on every render to avoid spam
      } catch (e) {
        console.error('[Game] Failed to cache state:', e);
      }
    }
  }); // No dependency array - runs after every render

  // Clear cache on mount (new game state has loaded)
  React.useEffect(() => {
    console.log('[Game] Mounted - will use cached state during transition');
    // Clear cache after a delay to allow transition to complete
    const timer = setTimeout(() => {
      sessionStorage.removeItem('gameStateCache');
      sessionStorage.removeItem('stageStateCache');
      console.log('[Game] Cleared all caches after transition complete');
    }, 200);
    return () => clearTimeout(timer);
  }, []);

  return (
    <div ref={containerRef} className="h-full w-full flex">
      <div className="h-full w-full flex flex-col">
      <Profile />
        <div className="h-full flex items-center justify-center">
          <Stage key="stable-stage" />
        </div>
      </div>
    </div>
  );
}

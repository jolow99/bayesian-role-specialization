import React from "react";

/**
 * Skeleton loading screen that displays cached game state during transitions
 * to minimize visual flash and maintain continuity
 */
export function SkeletonLoader() {
  // Try to retrieve cached game state from sessionStorage
  // Prefer stage cache (more specific) over game cache (for round transitions)
  const cachedState = React.useMemo(() => {
    try {
      // First try stage-level cache (for transitions within a round)
      let stored = sessionStorage.getItem('stageStateCache');
      if (stored) {
        console.log('[SkeletonLoader] Using stage cache');
        return JSON.parse(stored);
      }

      // Fallback to game-level cache (for round transitions)
      stored = sessionStorage.getItem('gameStateCache');
      if (stored) {
        console.log('[SkeletonLoader] Using game cache');
        return JSON.parse(stored);
      }

      console.log('[SkeletonLoader] No cache found');
      return null;
    } catch (e) {
      console.error('Failed to load cached game state:', e);
      return null;
    }
  }, []);

  // If we have cached state, render it without any overlay for seamless transition
  if (cachedState) {
    return (
      <div
        className="skeleton-loader-with-cache"
        dangerouslySetInnerHTML={{ __html: cachedState.html }}
      />
    );
  }

  // Fallback: Generic skeleton if no cached state available
  return (
    <div className="fixed inset-0 bg-gradient-to-b from-blue-400 to-blue-600 flex items-center justify-center p-2">
      <div className="w-full h-full flex items-center justify-center" style={{ maxWidth: '1400px' }}>
        {/* Battle Screen Container */}
        <div className="bg-white rounded-lg shadow-2xl border-4 border-gray-800 w-full h-full flex overflow-hidden relative">
          {/* Left Column - Game Interface */}
          <div className="flex-1 flex flex-col min-w-0">
            {/* Round Header Skeleton */}
            <div className="bg-gray-800 text-white text-center flex-shrink-0 rounded-tl-lg flex items-center justify-center" style={{ height: '40px' }}>
              <div className="h-5 w-32 bg-gray-600 rounded animate-pulse"></div>
            </div>

            {/* Battle Field Skeleton */}
            <div className="flex-shrink-0 bg-gradient-to-b from-gray-700 to-gray-900" style={{ height: '35vh', minHeight: '250px', maxHeight: '400px' }}>
              <div className="h-full flex flex-col items-center justify-center p-8">
                {/* Enemy Health Bar Skeleton */}
                <div className="w-full max-w-md mb-12">
                  <div className="h-3 bg-gray-600 rounded-full mb-2 animate-pulse"></div>
                  <div className="h-8 bg-gray-600 rounded animate-pulse"></div>
                </div>

                {/* Battle Area Placeholder */}
                <div className="flex-1 flex items-center justify-center">
                  <div className="w-32 h-32 bg-gray-600 rounded-full animate-pulse"></div>
                </div>

                {/* Team Health Bar Skeleton */}
                <div className="w-full max-w-md mt-12">
                  <div className="h-3 bg-gray-600 rounded-full mb-2 animate-pulse"></div>
                  <div className="h-8 bg-gray-600 rounded animate-pulse"></div>
                </div>
              </div>
            </div>

            {/* Action Area Skeleton */}
            <div className="bg-white border-t-4 border-gray-700 flex-1 min-h-0 flex flex-col">
              <div className="flex-1 p-4 flex items-center justify-center">
                <div className="w-full max-w-4xl">
                  {/* Role Selection Buttons Skeleton */}
                  <div className="flex gap-4 justify-center mb-6">
                    {[1, 2, 3].map((i) => (
                      <div key={i} className="flex-1 max-w-xs">
                        <div className="h-32 bg-gray-200 rounded-lg animate-pulse"></div>
                      </div>
                    ))}
                  </div>
                  {/* Submit Button Skeleton */}
                  <div className="flex justify-center">
                    <div className="h-12 w-48 bg-gray-200 rounded-lg animate-pulse"></div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column - Battle History Skeleton */}
          <div className="bg-gray-50 border-l-4 border-gray-700 overflow-hidden flex flex-col" style={{ width: '22%', minWidth: '280px', maxWidth: '350px' }}>
            <div className="bg-gray-800 text-white text-center flex-shrink-0 rounded-tr-lg flex items-center justify-center" style={{ height: '40px' }}>
              <div className="h-4 w-32 bg-gray-600 rounded animate-pulse"></div>
            </div>
            <div className="flex-1 overflow-auto p-3 bg-white">
              {/* History Entry Skeletons */}
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="mb-4 p-3 bg-gray-100 rounded animate-pulse">
                  <div className="h-4 bg-gray-300 rounded mb-2"></div>
                  <div className="h-3 bg-gray-300 rounded w-3/4"></div>
                </div>
              ))}
            </div>
          </div>

          {/* Loading Indicator Overlay */}
          <div className="absolute inset-0 bg-white bg-opacity-20 flex items-center justify-center pointer-events-none">
            <div className="bg-white bg-opacity-90 rounded-lg p-6 shadow-lg">
              <div className="flex items-center gap-3">
                <div className="w-6 h-6 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                <span className="text-gray-700 font-medium">Loading...</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

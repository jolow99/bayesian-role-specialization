import React, { useState, useEffect } from "react";
import { createPortal } from "react-dom";

/**
 * TutorialSpotlight creates a dark overlay with a "cut-out" window
 * to highlight specific elements on the page
 */
export function TutorialSpotlight({ targetRect, padding = 16, showBorder = true }) {
  const [highlightRect, setHighlightRect] = useState(null);

  useEffect(() => {
    if (targetRect) {
      setHighlightRect({
        top: Math.max(0, targetRect.top - padding),
        left: Math.max(0, targetRect.left - padding),
        width: targetRect.width + padding * 2,
        height: targetRect.height + padding * 2
      });
    }
  }, [targetRect, padding]);

  if (!highlightRect) return null;

  return createPortal(
    <div
      className="fixed inset-0 pointer-events-none transition-all duration-500"
      style={{ zIndex: 1000 }}
    >
      {/* Dark overlay with cut-out using clip-path */}
      <div
        className="absolute inset-0 bg-black transition-all duration-500"
        style={{
          opacity: 0.7,
          clipPath: createClipPath(highlightRect)
        }}
      />

      {/* Animated border around highlighted region */}
      {showBorder && (
        <div
          className="absolute border-4 border-blue-400 rounded-lg transition-all duration-500"
          style={{
            top: highlightRect.top,
            left: highlightRect.left,
            width: highlightRect.width,
            height: highlightRect.height,
            animation: "pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite"
          }}
        />
      )}

      {/* Add pulse animation */}
      <style>
        {`
          @keyframes pulse {
            0%, 100% {
              opacity: 1;
            }
            50% {
              opacity: 0.5;
            }
          }
        `}
      </style>
    </div>,
    document.body
  );
}

/**
 * Creates a CSS clip-path that shows everything except the highlighted rectangle
 * This creates the "cut-out" effect
 */
function createClipPath(rect) {
  const { top, left, width, height } = rect;
  const right = left + width;
  const bottom = top + height;

  // Create a polygon that covers the entire viewport except the highlighted area
  // The polygon goes: outer rectangle (clockwise) -> inner rectangle (counter-clockwise)
  return `polygon(
    0% 0%,
    0% 100%,
    ${left}px 100%,
    ${left}px ${top}px,
    ${right}px ${top}px,
    ${right}px ${bottom}px,
    ${left}px ${bottom}px,
    ${left}px 100%,
    100% 100%,
    100% 0%
  )`;
}

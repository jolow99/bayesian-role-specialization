import React, { useState, useEffect } from "react";
import { createPortal } from "react-dom";

/**
 * TutorialSpotlight creates a dark overlay with a "cut-out" window
 * to highlight specific elements on the page
 */
export function TutorialSpotlight({ targetRects = [], padding = 16, showBorder = true }) {
  const [highlightRects, setHighlightRects] = useState([]);

  useEffect(() => {
    if (targetRects && targetRects.length > 0) {
      // Process all rects synchronously
      const rects = targetRects.map(targetRect => ({
        top: Math.max(0, targetRect.top - padding),
        left: Math.max(0, targetRect.left - padding),
        width: targetRect.width + padding * 2,
        height: targetRect.height + padding * 2
      }));
      setHighlightRects(rects);
    } else {
      setHighlightRects([]);
    }
  }, [targetRects, padding]);

  if (highlightRects.length === 0) return null;

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
          clipPath: createClipPath(highlightRects)
        }}
      />

      {/* Animated borders around all highlighted regions */}
      {showBorder && highlightRects.map((rect, index) => (
        <div
          key={index}
          className="absolute border-4 border-blue-400 rounded-lg transition-all duration-500"
          style={{
            top: rect.top,
            left: rect.left,
            width: rect.width,
            height: rect.height,
            animation: "pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite"
          }}
        />
      ))}

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
 * Creates a CSS clip-path that shows everything except the highlighted rectangles
 * This creates the "cut-out" effect for multiple regions
 */
function createClipPath(rects) {
  if (rects.length === 0) {
    return 'polygon(0% 0%, 0% 100%, 100% 100%, 100% 0%)';
  }

  if (rects.length === 1) {
    const { top, left, width, height } = rects[0];
    const right = left + width;
    const bottom = top + height;

    // Single rectangle path
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

  // Multiple rectangles - create multiple cut-outs
  // Start with outer border
  let path = '0% 0%, 0% 100%';

  // Add each rectangle as a cut-out
  rects.forEach(rect => {
    const { top, left, width, height } = rect;
    const right = left + width;
    const bottom = top + height;

    path += `, ${left}px 100%, ${left}px ${top}px, ${right}px ${top}px, ${right}px ${bottom}px, ${left}px ${bottom}px, ${left}px 100%`;
  });

  // Complete the outer border
  path += ', 100% 100%, 100% 0%';

  return `polygon(${path})`;
}

import React, { useState, useEffect, useRef } from "react";
import { Button } from "../Button";

/**
 * TutorialTooltip displays explanatory text next to highlighted elements
 * with auto-positioning to stay on screen
 */
export function TutorialTooltip({
  content,
  position = "right", // preferred position: top, right, bottom, left, center
  targetRect,
  onNext,
  onBack,
  onSkip,
  onReplay,
  stepNumber,
  totalSteps,
  showNext = true,
  showSkip = true
}) {
  const tooltipRef = useRef(null);
  const [calculatedPosition, setCalculatedPosition] = useState({ top: 0, left: 0, position });
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    if (!tooltipRef.current || !targetRect) return;

    const tooltip = tooltipRef.current;
    const tooltipRect = tooltip.getBoundingClientRect();
    const padding = 20; // Space between tooltip and target

    const positions = calculatePositions(targetRect, tooltipRect, padding);
    const bestPosition = findBestPosition(positions, tooltipRect, position);

    setCalculatedPosition(bestPosition);
    setIsVisible(true);
  }, [targetRect, position]);


  return (
    <div
      ref={tooltipRef}
      role="dialog"
      aria-labelledby="tutorial-title"
      aria-describedby="tutorial-content"
      tabIndex={-1}
      className={`fixed bg-white rounded-lg shadow-2xl border-4 border-blue-500 p-6 max-w-md transition-opacity duration-300 ${
        isVisible ? "opacity-100" : "opacity-0"
      }`}
      style={{
        top: calculatedPosition.top,
        left: calculatedPosition.left,
        zIndex: 1002,
        maxHeight: "80vh",
        overflowY: "auto"
      }}
    >
      {/* Step counter */}
      {stepNumber !== undefined && totalSteps !== undefined && (
        <div className="text-xs font-semibold text-blue-600 mb-2">
          Step {stepNumber} of {totalSteps}
        </div>
      )}

      {/* Content */}
      <div id="tutorial-content" className="text-gray-800 mb-4">
        {typeof content === "string" ? (
          <p className="text-sm leading-relaxed">{content}</p>
        ) : (
          content
        )}
      </div>

      {/* Navigation buttons */}
      <div className="flex gap-3 justify-between items-center">
        {/* Left side - Skip and Back buttons */}
        <div className="flex gap-3">
          {showSkip && onSkip && (
            <button
              onClick={onSkip}
              className="px-4 py-2 text-sm text-gray-600 hover:text-gray-800 transition-colors"
            >
              Skip Tutorial
            </button>
          )}
          {onBack && (
            <Button handleClick={onBack}>
              <p>Back</p>
            </Button>
          )}
        </div>

        {/* Right side - Next/Finish and Replay buttons */}
        <div className="flex gap-3">
          {onReplay && (
            <Button handleClick={onReplay}>
              <p>Replay</p>
            </Button>
          )}
          {showNext && onNext && (
            <Button handleClick={onNext} autoFocus>
              <p>{stepNumber === totalSteps ? "Finish" : "Next"}</p>
            </Button>
          )}
        </div>
      </div>

      {/* Arrow pointer */}
      {calculatedPosition.position !== "center" && (
        <div
          className="absolute w-0 h-0 border-solid"
          style={getArrowStyle(calculatedPosition.position)}
        />
      )}
    </div>
  );
}

/**
 * Calculate all possible positions for the tooltip
 */
function calculatePositions(targetRect, tooltipRect, padding) {
  return {
    right: {
      top: targetRect.top + (targetRect.height - tooltipRect.height) / 2,
      left: targetRect.right + padding,
      position: "right"
    },
    left: {
      top: targetRect.top + (targetRect.height - tooltipRect.height) / 2,
      left: targetRect.left - tooltipRect.width - padding,
      position: "left"
    },
    top: {
      top: targetRect.top - tooltipRect.height - padding,
      left: targetRect.left + (targetRect.width - tooltipRect.width) / 2,
      position: "top"
    },
    bottom: {
      top: targetRect.bottom + padding,
      left: targetRect.left + (targetRect.width - tooltipRect.width) / 2,
      position: "bottom"
    },
    center: {
      top: (window.innerHeight - tooltipRect.height) / 2,
      left: (window.innerWidth - tooltipRect.width) / 2,
      position: "center"
    }
  };
}

/**
 * Find the best position that fits on screen
 */
function findBestPosition(positions, tooltipRect, preferredPosition) {
  // Try preferred position first
  if (positions[preferredPosition] && fitsOnScreen(positions[preferredPosition], tooltipRect)) {
    return positions[preferredPosition];
  }

  // Try other positions in order of preference
  const fallbackOrder = ["right", "left", "bottom", "top", "center"];
  for (const pos of fallbackOrder) {
    if (positions[pos] && fitsOnScreen(positions[pos], tooltipRect)) {
      return positions[pos];
    }
  }

  // Fallback to center
  return positions.center;
}

/**
 * Check if tooltip fits on screen at given position
 */
function fitsOnScreen(pos, tooltipRect) {
  if (pos.position === "center") return true;

  const margin = 10;
  return (
    pos.top >= margin &&
    pos.left >= margin &&
    pos.top + tooltipRect.height <= window.innerHeight - margin &&
    pos.left + tooltipRect.width <= window.innerWidth - margin
  );
}

/**
 * Get CSS styles for arrow pointer based on position
 */
function getArrowStyle(position) {
  const arrowSize = 10;

  const styles = {
    right: {
      left: -arrowSize,
      top: "50%",
      transform: "translateY(-50%)",
      borderWidth: `${arrowSize}px ${arrowSize}px ${arrowSize}px 0`,
      borderColor: `transparent #3b82f6 transparent transparent`
    },
    left: {
      right: -arrowSize,
      top: "50%",
      transform: "translateY(-50%)",
      borderWidth: `${arrowSize}px 0 ${arrowSize}px ${arrowSize}px`,
      borderColor: `transparent transparent transparent #3b82f6`
    },
    top: {
      bottom: -arrowSize,
      left: "50%",
      transform: "translateX(-50%)",
      borderWidth: `${arrowSize}px ${arrowSize}px 0 ${arrowSize}px`,
      borderColor: `#3b82f6 transparent transparent transparent`
    },
    bottom: {
      top: -arrowSize,
      left: "50%",
      transform: "translateX(-50%)",
      borderWidth: `0 ${arrowSize}px ${arrowSize}px ${arrowSize}px`,
      borderColor: `transparent transparent #3b82f6 transparent`
    }
  };

  return styles[position] || {};
}

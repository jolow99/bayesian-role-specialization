import React, { useState, useEffect, useRef } from "react";
import { TutorialSpotlight } from "./TutorialSpotlight";
import { TutorialTooltip } from "./TutorialTooltip";

/**
 * TutorialWrapper orchestrates the step-by-step guided tour
 *
 * @param {Array} steps - Array of step configurations:
 *   {
 *     targetId: "data-tutorial-id value" or null for full screen,
 *     content: "Explanation text" or JSX element,
 *     tooltipPosition: "right" | "left" | "top" | "bottom" | "center",
 *     onEnter: optional callback when entering this step
 *   }
 * @param {Function} onComplete - Called when tutorial is finished
 * @param {Function} onSkip - Called when tutorial is skipped
 * @param {ReactNode} children - The interface to wrap with tutorial
 */
export function TutorialWrapper({ steps = [], onComplete, onSkip, children }) {
  const [currentStep, setCurrentStep] = useState(0);
  const [targetRect, setTargetRect] = useState(null);
  const wrapperRef = useRef(null);

  // Find and measure the target element whenever step changes
  useEffect(() => {
    if (currentStep >= steps.length) {
      // Tutorial complete
      if (onComplete) onComplete();
      return;
    }

    const step = steps[currentStep];

    // Call onEnter callback if provided
    if (step.onEnter) {
      step.onEnter();
    }

    // Wait for next tick to ensure DOM is ready
    const timer = setTimeout(() => {
      if (!step.targetId) {
        // Highlight entire wrapper
        if (wrapperRef.current) {
          const rect = wrapperRef.current.getBoundingClientRect();
          setTargetRect(rect);
        }
      } else {
        // Find element by data-tutorial-id
        const element = document.querySelector(`[data-tutorial-id="${step.targetId}"]`);
        if (element) {
          const rect = element.getBoundingClientRect();
          setTargetRect(rect);
        } else {
          console.warn(`[Tutorial] Could not find element with data-tutorial-id="${step.targetId}"`);
          // Fallback to wrapper
          if (wrapperRef.current) {
            const rect = wrapperRef.current.getBoundingClientRect();
            setTargetRect(rect);
          }
        }
      }
    }, 100);

    return () => clearTimeout(timer);
  }, [currentStep, steps, onComplete]);

  // Handle window resize to update target rect
  useEffect(() => {
    const handleResize = () => {
      if (currentStep >= steps.length) return;

      const step = steps[currentStep];
      if (!step.targetId && wrapperRef.current) {
        const rect = wrapperRef.current.getBoundingClientRect();
        setTargetRect(rect);
      } else if (step.targetId) {
        const element = document.querySelector(`[data-tutorial-id="${step.targetId}"]`);
        if (element) {
          const rect = element.getBoundingClientRect();
          setTargetRect(rect);
        }
      }
    };

    // Debounce resize events
    let resizeTimer;
    const debouncedResize = () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(handleResize, 150);
    };

    window.addEventListener("resize", debouncedResize);
    return () => {
      window.removeEventListener("resize", debouncedResize);
      clearTimeout(resizeTimer);
    };
  }, [currentStep, steps]);

  const handleNext = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      // Last step, complete tutorial
      if (onComplete) onComplete();
    }
  };

  // Don't show tutorial if no steps
  if (steps.length === 0) {
    return <div ref={wrapperRef}>{children}</div>;
  }

  // Tutorial is complete
  if (currentStep >= steps.length) {
    return <div ref={wrapperRef}>{children}</div>;
  }

  const currentStepConfig = steps[currentStep];

  return (
    <div ref={wrapperRef} className="relative">
      {children}

      {/* Spotlight overlay */}
      {targetRect && (
        <TutorialSpotlight
          targetRect={targetRect}
          padding={16}
          showBorder={true}
        />
      )}

      {/* Tooltip */}
      {targetRect && (
        <TutorialTooltip
          content={currentStepConfig.content}
          position={currentStepConfig.tooltipPosition || "right"}
          targetRect={targetRect}
          onNext={handleNext}
          onSkip={onSkip}
          stepNumber={currentStep + 1}
          totalSteps={steps.length}
          showNext={true}
          showSkip={currentStep === 0 && !!onSkip} // Only show skip on first step if onSkip handler provided
        />
      )}
    </div>
  );
}

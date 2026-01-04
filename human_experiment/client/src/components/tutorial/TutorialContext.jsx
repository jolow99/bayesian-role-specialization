import React, { createContext, useContext } from "react";

// Create context for tutorial mode
const TutorialContext = createContext({
  isTutorialMode: false,
  mockData: null
});

// Provider component
export function TutorialProvider({ children, mockData }) {
  return (
    <TutorialContext.Provider value={{ isTutorialMode: true, mockData }}>
      {children}
    </TutorialContext.Provider>
  );
}

// Hook to use tutorial context
export function useTutorialContext() {
  return useContext(TutorialContext);
}

export default TutorialContext;

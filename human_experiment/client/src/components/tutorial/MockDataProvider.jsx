import React from "react";
import { TutorialProvider } from "./TutorialContext";

/**
 * MockDataProvider wraps tutorial components and provides mock data
 * that mimics the Empirica hook structure. Components can then use
 * TutorialContext to check if they're in tutorial mode and use mock data.
 */
export function MockDataProvider({ children, mockData }) {
  // Structure mock data to match Empirica's get/set pattern
  const structuredMockData = {
    game: createMockEntity(mockData.game || {}),
    player: createMockPlayer(mockData.player || {}),
    players: (mockData.players || []).map(p => createMockPlayer(p)),
    round: createMockEntity(mockData.round || {}),
    stage: createMockEntity(mockData.stage || {}),
    // Additional data
    teamHistory: mockData.teamHistory || [],
    virtualBots: mockData.virtualBots || []
  };

  return (
    <TutorialProvider mockData={structuredMockData}>
      {children}
    </TutorialProvider>
  );
}

/**
 * Creates a mock entity with get/set methods that mimic Empirica's pattern
 */
function createMockEntity(data) {
  return {
    get: (key) => data[key],
    set: (key, value) => {
      // In tutorial mode, sets are no-ops (we don't persist)
      console.log(`[Tutorial] Mock set: ${key} = ${value}`);
    },
    ...data // Spread data for direct property access if needed
  };
}

/**
 * Creates a mock player with nested stage and round entities
 */
function createMockPlayer(playerData) {
  const player = {
    id: playerData.id || "tutorial-player",
    playerId: playerData.playerId,
    stats: playerData.stats,
    get: (key) => playerData[key],
    set: (key, value) => {
      console.log(`[Tutorial] Mock player set: ${key} = ${value}`);
    },
    stage: {
      get: (key) => playerData.stage?.[key],
      set: (key, value) => {
        console.log(`[Tutorial] Mock player.stage set: ${key} = ${value}`);
      }
    },
    round: {
      get: (key) => playerData.round?.[key],
      set: (key, value) => {
        console.log(`[Tutorial] Mock player.round set: ${key} = ${value}`);
      }
    },
    ...playerData
  };

  return player;
}

/**
 * Helper function to create default mock data for tutorials
 */
export function createDefaultMockData() {
  return {
    game: {
      enemyHealth: 30,
      maxEnemyHealth: 30,
      teamHealth: 15,
      maxTeamHealth: 15,
      treatment: {
        totalPlayers: 3,
        maxRounds: 5,
        maxEnemyHealth: 30,
        maxTeamHealth: 15
      },
      virtualBots: []
    },
    player: {
      id: "tutorial-player",
      playerId: 0,
      stats: { STR: 2, DEF: 2, SUP: 2 },
      roleOrder: [0, 1, 2], // Fighter, Tank, Medic
      stage: {},
      round: {}
    },
    players: [
      {
        id: "tutorial-player",
        playerId: 0,
        stats: { STR: 2, DEF: 2, SUP: 2 }
      },
      {
        id: "bot-1",
        playerId: 1,
        stats: { STR: 3, DEF: 1, SUP: 2 }
      },
      {
        id: "bot-2",
        playerId: 2,
        stats: { STR: 1, DEF: 3, SUP: 2 }
      }
    ],
    round: {
      roundNumber: 1
    },
    stage: {
      name: "roleSelection",
      stageType: "roleSelection"
    },
    teamHistory: []
  };
}

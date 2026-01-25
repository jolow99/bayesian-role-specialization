import React, { useState, useEffect } from "react";
import { MockDataProvider, TutorialWrapper } from "../components/tutorial";
import { BattleField } from "../components/BattleField";
import { ActionMenu } from "../components/ActionMenu";
import { ResultsPanel } from "../components/ResultsPanel";
import { ActionHistory } from "../components/ActionHistory";
import { ROLES, ROLE_ICONS, ROLE_NAMES, ROLE_LABELS } from "../constants";

export function Tutorial2({ next }) {
  const [selectedRole, setSelectedRole] = useState(null);
  const [inferredRoles, setInferredRoles] = useState({}); // { playerId: roleValue } - user's inference of other players' roles
  const [mockData, setMockData] = useState(null);
  const [showOutcome, setShowOutcome] = useState(false);
  const [outcome, setOutcome] = useState(null);
  const [showDamageAnimation, setShowDamageAnimation] = useState(false);
  const [round1Turn1Result, setRound1Turn1Result] = useState(null);
  const [round1Turn2Result, setRound1Turn2Result] = useState(null);
  const [allStageResults, setAllStageResults] = useState([]); // Array of { stageNum, role, turns: [turn1, turn2] }
  const [currentStageIndex, setCurrentStageIndex] = useState(0); // Which stage we're viewing (0-indexed, stage 2 = index 0)
  const [currentTurnInStage, setCurrentTurnInStage] = useState(0); // 0 = role selection, 1 = turn 1, 2 = turn 2
  const [currentGameState, setCurrentGameState] = useState("initial"); // initial, round1-turn1, round1-turn2, role-selection, stage-turn, outcome
  const [introTutorialComplete, setIntroTutorialComplete] = useState(false);
  const [roleSelectionTutorialComplete, setRoleSelectionTutorialComplete] = useState(false);
  const [roleHistory, setRoleHistory] = useState([]); // Track roles chosen for each stage

  // Bot players: One Tank (blocks when enemy attacks), One MEDIC (heals when health < 100%)
  const actualBotRoles = [ROLES.TANK, ROLES.MEDIC];

  // Intro tutorial steps - shown at the start before watching Stage 1
  const introTutorialSteps = [
    {
      targetId: "full-screen",
      tooltipPosition: "center",
      showBorder: false,
      content: (
        <div>
          <h3 className="text-lg font-bold text-gray-900 mb-3">Tutorial Game 2</h3>
          <p className="text-sm text-gray-700 mb-3">
            In this tutorial, you'll learn how to analyze battle patterns and choose the best role to complement your team.
          </p>
          <p className="text-sm text-gray-700">
            Unlike Tutorial 1, you won't see your teammates' roles directly. Instead, you'll need to infer them from their actions!
          </p>
        </div>
      )
    },
    {
      targetId: "full-screen",
      tooltipPosition: "center",
      showBorder: false,
      content: (
        <div>
          <h4 className="text-lg font-bold text-gray-900 mb-3">How It Works</h4>
          <p className="text-sm text-gray-700 mb-3">
            Stage 1 has already been played by your two teammates. You'll watch what happened, then join them for Stage 2.
          </p>
          <p className="text-sm text-gray-700 font-semibold">
            Pay attention to what actions each player took — this will help you figure out their roles!
          </p>
        </div>
      )
    }
  ];

  // Role selection tutorial step - shown when reaching role selection
  const roleSelectionTutorialSteps = [
    {
      targetId: "inference-section",
      tooltipPosition: "top",
      content: (
        <div>
          <h4 className="text-lg font-bold text-gray-900 mb-2">Report Your Inferences</h4>
          <p className="text-sm text-gray-700 mb-2">
            First, report what roles you think P1 and P2 have based on their actions.
          </p>
          <p className="text-sm text-gray-700">
            Look at the battle history: P1 blocked when attacked (Tank?), P2 healed when damaged (Medic?).
          </p>
        </div>
      )
    },
    {
      targetId: "action-menu",
      tooltipPosition: "top",
      content: (
        <div>
          <h4 className="text-lg font-bold text-gray-900 mb-2">Choose Your Role</h4>
          <p className="text-sm text-gray-700 mb-2">
            Now select a role to complement your team for Stage 2.
          </p>
          <p className="text-sm text-gray-700 font-semibold">
            Based on what you inferred, which role would best help the team?
          </p>
        </div>
      )
    }
  ];

  const handleIntroTutorialComplete = () => {
    setIntroTutorialComplete(true);
    // Transition to Turn 1 results after intro tutorial
    if (round1Turn1Result) {
      const turn1MockData = createMockDataForRound1Turn1(round1Turn1Result);
      setMockData(turn1MockData);
      setCurrentGameState("round1-turn1");
      setShowDamageAnimation(true);
      setTimeout(() => setShowDamageAnimation(false), 2000);
    }
  };

  const handleRoleSelectionTutorialComplete = () => {
    setRoleSelectionTutorialComplete(true);
  };

  useEffect(() => {
    // Initialize Stage 1 on mount
    initializeRound1();
  }, []);


  const getBotAction = (role, enemyAttacks, teamHealth) => {
    if (role === ROLES.FIGHTER) return "ATTACK";
    if (role === ROLES.TANK) return enemyAttacks ? "BLOCK" : "ATTACK";
    if (role === ROLES.MEDIC) {
      // MEDIC heals when team health < 100% 
      return teamHealth < 10 ? "HEAL" : "ATTACK";
    }
    return "ATTACK";
  };

  // Simulate a single turn of combat using real game stats
  const simulateTurn = (turnNum, playerRole, currentEnemyHP, currentTeamHP, enemyAttacks) => {
    const enemyIntent = enemyAttacks ? "WILL_ATTACK" : "WILL_REST";
    const STR = 2, DEF = 2, SUP = 2; // Real game stats
    const bossDamage = 6; // Tutorial 2 boss damage
    const maxTeamHealth = 10;

    // Calculate bot actions
    const bot1Action = getBotAction(actualBotRoles[0], enemyAttacks, currentTeamHP);
    const bot2Action = getBotAction(actualBotRoles[1], enemyAttacks, currentTeamHP);

    let playerAction, actions, roles, stats;
    if (playerRole === null) {
      // Player hasn't chosen yet - only 2 bots
      playerAction = null;
      actions = [bot1Action, bot2Action];
      roles = actualBotRoles;
      stats = [{ STR, DEF, SUP }, { STR, DEF, SUP }];
    } else {
      // Player has chosen - calculate their action
      playerAction = getBotAction(playerRole, enemyAttacks, currentTeamHP);
      actions = [bot1Action, bot2Action, playerAction];
      roles = [...actualBotRoles, playerRole];
      stats = [{ STR, DEF, SUP }, { STR, DEF, SUP }, { STR, DEF, SUP }];
    }

    // Calculate damage using real game logic (additive STR)
    let totalAttack = 0;
    actions.forEach((action, idx) => {
      if (action === "ATTACK") {
        totalAttack += stats[idx].STR;
      }
    });
    const damageToEnemy = totalAttack;

    // Calculate defense using real game logic (max DEF, not additive)
    let maxDefense = 0;
    actions.forEach((action, idx) => {
      if (action === "BLOCK") {
        maxDefense = Math.max(maxDefense, stats[idx].DEF);
      }
    });

    let damageToTeam = 0;
    let damageBlocked = 0;
    if (enemyAttacks) {
      const mitigatedDamage = bossDamage - maxDefense;
      damageToTeam = Math.max(0, mitigatedDamage);
      damageBlocked = Math.min(maxDefense, bossDamage);
    }

    // Calculate healing using real game logic (additive SUP)
    let totalHeal = 0;
    actions.forEach((action, idx) => {
      if (action === "HEAL") {
        totalHeal += stats[idx].SUP;
      }
    });
    const healAmount = totalHeal;

    const newEnemyHP = Math.max(0, currentEnemyHP - damageToEnemy);
    const newTeamHP = Math.max(0, Math.min(maxTeamHealth, currentTeamHP - damageToTeam + healAmount));

    return {
      turnNum,
      enemyAttacks,
      enemyIntent,
      bot1Action,
      bot2Action,
      playerAction,
      playerRole,
      actions,
      roles,
      damageToEnemy: Math.round(damageToEnemy),
      damageToTeam: Math.round(damageToTeam),
      damageBlocked: Math.round(damageBlocked),
      healAmount: Math.round(healAmount),
      enemyHealth: Math.round(newEnemyHP),
      teamHealth: Math.round(newTeamHP),
      previousEnemyHealth: Math.round(currentEnemyHP),
      previousTeamHealth: Math.round(currentTeamHP)
    };
  };

  const initializeRound1 = () => {
    // Calculate both turns but don't show them yet
    const r1t1 = simulateTurn(1, null, 10, 10, true);
    const r1t2 = simulateTurn(2, null, r1t1.enemyHealth, r1t1.teamHealth, false);

    setRound1Turn1Result(r1t1);
    setRound1Turn2Result(r1t2);

    // Start at initial state (empty game interface)
    const initialMockData = createMockDataForInitial();
    setMockData(initialMockData);
    setCurrentGameState("initial");
  };

  const createMockDataForInitial = () => {
    const players = [
      { id: "bot-1", playerId: 0, stats: { STR: 2, DEF: 2, SUP: 2 } },
      { id: "bot-2", playerId: 1, stats: { STR: 2, DEF: 2, SUP: 2 } }
    ];

    const roundData = {
      roundNumber: 1,
      stageNumber: 0
    };

    return {
      game: {
        enemyHealth: 10,
        maxEnemyHealth: 10,
        teamHealth: 10,
        maxTeamHealth: 10,
        treatment: {
          totalPlayers: 2,
          maxRounds: 2,
          maxEnemyHealth: 10,
          maxTeamHealth: 10
        },
        get: (key) => {
          if (key === "round1Config") {
            return { maxTeamHealth: 10, maxEnemyHealth: 10 };
          }
          return undefined;
        }
      },
      player: {
        id: "tutorial-player",
        playerId: 2,
        stats: { STR: 2, DEF: 2, SUP: 2 },
        roleOrder: [ROLES.FIGHTER, ROLES.TANK, ROLES.MEDIC],
        stage: {},
        round: {},
        get: (key) => {
          if (key === "actionHistory") return [];
          if (key === "gamePlayerId") return 2;
          return undefined;
        }
      },
      players: players,
      round: {
        ...roundData,
        get: (key) => roundData[key]
      },
      stage: {
        name: "initial",
        stageType: "initial",
        turnNumber: 0
      }
    };
  };

  const createMockDataForRound1Turn1 = (turn1Result) => {
    const players = [
      { id: "bot-1", playerId: 0, stats: { STR: 2, DEF: 2, SUP: 2 } },
      { id: "bot-2", playerId: 1, stats: { STR: 2, DEF: 2, SUP: 2 } }
    ];

    const roundData = {
      roundNumber: 1,
      stageNumber: 1,
      stage1Turns: [{
        turnNumber: 1,
        enemyIntent: turn1Result.enemyIntent,
        actions: turn1Result.actions,
        damageToEnemy: turn1Result.damageToEnemy,
        damageToTeam: turn1Result.damageToTeam,
        damageBlocked: turn1Result.damageBlocked,
        healAmount: turn1Result.healAmount,
        previousEnemyHealth: 10,
        previousTeamHealth: 10,
        newEnemyHealth: turn1Result.enemyHealth,
        newTeamHealth: turn1Result.teamHealth
      }]
    };

    return {
      game: {
        enemyHealth: turn1Result.enemyHealth,
        maxEnemyHealth: 10,
        teamHealth: turn1Result.teamHealth,
        maxTeamHealth: 10,
        treatment: {
          totalPlayers: 2,
          maxRounds: 2,
          maxEnemyHealth: 10,
          maxTeamHealth: 10
        },
        get: (key) => {
          if (key === "round1Config") {
            return { maxTeamHealth: 10, maxEnemyHealth: 10 };
          }
          return undefined;
        }
      },
      player: {
        id: "tutorial-player",
        playerId: 2,
        stats: { STR: 2, DEF: 2, SUP: 2 },
        roleOrder: [ROLES.FIGHTER, ROLES.TANK, ROLES.MEDIC],
        stage: {},
        round: {},
        get: (key) => {
          if (key === "actionHistory") return [];
          if (key === "gamePlayerId") return 2;
          return undefined;
        }
      },
      players: players,
      round: {
        ...roundData,
        get: (key) => roundData[key]
      },
      stage: {
        name: "turn1",
        stageType: "turn",
        turnNumber: 1
      }
    };
  };

  const showRound1Turn2 = () => {
    if (!round1Turn1Result || !round1Turn2Result) return;

    const newMockData = createMockDataForRound1Complete(round1Turn1Result, round1Turn2Result);
    setMockData(newMockData);
    setCurrentGameState("round1-turn2");
    setShowDamageAnimation(true);
    setTimeout(() => setShowDamageAnimation(false), 2000);
  };

  const completeRound1 = () => {
    // Go directly to role selection
    const newMockData = createMockDataForRoleSelection();
    setMockData(newMockData);
    setCurrentGameState("role-selection");
  };

  const createMockDataForRound1Complete = (turn1Result, turn2Result) => {
    const players = [
      { id: "bot-1", playerId: 0, stats: { STR: 2, DEF: 2, SUP: 2 } },
      { id: "bot-2", playerId: 1, stats: { STR: 2, DEF: 2, SUP: 2 } }
    ];

    const roundData = {
      roundNumber: 1,
      stageNumber: 1,
      stage1Turns: [
        {
          turnNumber: 1,
          enemyIntent: turn1Result.enemyIntent,
          actions: turn1Result.actions,
          damageToEnemy: turn1Result.damageToEnemy,
          damageToTeam: turn1Result.damageToTeam,
          damageBlocked: turn1Result.damageBlocked,
          healAmount: turn1Result.healAmount,
          previousEnemyHealth: 10,
          previousTeamHealth: 10,
          newEnemyHealth: turn1Result.enemyHealth,
          newTeamHealth: turn1Result.teamHealth
        },
        {
          turnNumber: 2,
          enemyIntent: turn2Result.enemyIntent,
          actions: turn2Result.actions,
          damageToEnemy: turn2Result.damageToEnemy,
          damageToTeam: turn2Result.damageToTeam,
          damageBlocked: turn2Result.damageBlocked,
          healAmount: turn2Result.healAmount,
          previousEnemyHealth: turn1Result.enemyHealth,
          previousTeamHealth: turn1Result.teamHealth,
          newEnemyHealth: turn2Result.enemyHealth,
          newTeamHealth: turn2Result.teamHealth
        }
      ]
    };

    return {
      game: {
        enemyHealth: turn2Result.enemyHealth,
        maxEnemyHealth: 10,
        teamHealth: turn2Result.teamHealth,
        maxTeamHealth: 10,
        treatment: {
          totalPlayers: 2,
          maxRounds: 2,
          maxEnemyHealth: 10,
          maxTeamHealth: 10
        },
        get: (key) => {
          if (key === "round1Config") {
            return { maxTeamHealth: 10, maxEnemyHealth: 10 };
          }
          return undefined;
        }
      },
      player: {
        id: "tutorial-player",
        playerId: 2,
        stats: { STR: 2, DEF: 2, SUP: 2 },
        roleOrder: [ROLES.FIGHTER, ROLES.TANK, ROLES.MEDIC],
        stage: {},
        round: {},
        get: (key) => {
          if (key === "actionHistory") return [];
          if (key === "gamePlayerId") return 2;
          return undefined;
        }
      },
      players: players,
      round: {
        ...roundData,
        get: (key) => roundData[key]
      },
      stage: {
        name: "turn2",
        stageType: "turn",
        turnNumber: 2
      }
    };
  };

  const createMockDataForRoleSelection = () => {
    const players = [
      { id: "bot-1", playerId: 0, stats: { STR: 2, DEF: 2, SUP: 2 } },
      { id: "bot-2", playerId: 1, stats: { STR: 2, DEF: 2, SUP: 2 } },
      { id: "tutorial-player", playerId: 2, stats: { STR: 2, DEF: 2, SUP: 2 } }
    ];

    // Get current health from last completed stage
    let currentEnemyHP, currentTeamHP;
    if (allStageResults.length === 0) {
      currentEnemyHP = round1Turn2Result.enemyHealth;
      currentTeamHP = round1Turn2Result.teamHealth;
    } else {
      const lastStage = allStageResults[allStageResults.length - 1];
      const lastTurn = lastStage.turns[lastStage.turns.length - 1];
      currentEnemyHP = lastTurn.enemyHealth;
      currentTeamHP = lastTurn.teamHealth;
    }

    // Next stage number
    const nextStageNum = allStageResults.length + 2;

    const roundData = {
      roundNumber: 1,
      stageNumber: nextStageNum,
      stage1Turns: [
        {
          turnNumber: 1,
          enemyIntent: round1Turn1Result.enemyIntent,
          actions: round1Turn1Result.actions,
          damageToEnemy: round1Turn1Result.damageToEnemy,
          damageToTeam: round1Turn1Result.damageToTeam,
          damageBlocked: round1Turn1Result.damageBlocked,
          healAmount: round1Turn1Result.healAmount,
          previousEnemyHealth: 10,
          previousTeamHealth: 10,
          newEnemyHealth: round1Turn1Result.enemyHealth,
          newTeamHealth: round1Turn1Result.teamHealth
        },
        {
          turnNumber: 2,
          enemyIntent: round1Turn2Result.enemyIntent,
          actions: round1Turn2Result.actions,
          damageToEnemy: round1Turn2Result.damageToEnemy,
          damageToTeam: round1Turn2Result.damageToTeam,
          damageBlocked: round1Turn2Result.damageBlocked,
          healAmount: round1Turn2Result.healAmount,
          previousEnemyHealth: round1Turn1Result.enemyHealth,
          previousTeamHealth: round1Turn1Result.teamHealth,
          newEnemyHealth: round1Turn2Result.enemyHealth,
          newTeamHealth: round1Turn2Result.teamHealth
        }
      ]
    };

    // Add all completed stage turns to roundData
    allStageResults.forEach((stageData) => {
      const stageKey = `stage${stageData.stageNum}Turns`;
      roundData[stageKey] = stageData.turns.map((result, idx) => ({
        turnNumber: idx + 1,
        enemyIntent: result.enemyIntent,
        actions: result.actions,
        damageToEnemy: result.damageToEnemy,
        damageToTeam: result.damageToTeam,
        damageBlocked: result.damageBlocked,
        healAmount: result.healAmount,
        previousEnemyHealth: result.previousEnemyHealth,
        previousTeamHealth: result.previousTeamHealth,
        newEnemyHealth: result.enemyHealth,
        newTeamHealth: result.teamHealth
      }));
    });

    // Build player action history showing roles for completed stages
    // Convert role number to string name (e.g., 0 -> "FIGHTER") for ActionHistory component
    const playerActionHistory = allStageResults.map((stageData) => ({
      round: 1,
      stage: stageData.stageNum,
      role: ROLE_NAMES[stageData.role]
    }));

    return {
      game: {
        enemyHealth: currentEnemyHP,
        maxEnemyHealth: 10,
        teamHealth: currentTeamHP,
        maxTeamHealth: 10,
        treatment: {
          totalPlayers: 3,
          maxRounds: 2,
          maxEnemyHealth: 10,
          maxTeamHealth: 10
        },
        get: (key) => {
          if (key === "round1Config") {
            return { maxTeamHealth: 10, maxEnemyHealth: 10 };
          }
          return undefined;
        }
      },
      player: {
        id: "tutorial-player",
        playerId: 2,
        stats: { STR: 2, DEF: 2, SUP: 2 },
        roleOrder: [ROLES.FIGHTER, ROLES.TANK, ROLES.MEDIC],
        stage: {},
        round: {
          get: (key) => {
            if (key === "playerId") return 2;
            return undefined;
          }
        },
        get: (key) => {
          if (key === "actionHistory") return playerActionHistory;
          if (key === "gamePlayerId") return 2;
          return undefined;
        }
      },
      players: players,
      round: {
        ...roundData,
        get: (key) => roundData[key]
      },
      stage: {
        name: "roleSelection",
        stageType: "roleSelection",
        stageNumber: nextStageNum
      }
    };
  };

  const createMockDataForStage = (stageResults, stageIdx, turnInStage) => {
    const players = [
      { id: "bot-1", playerId: 0, stats: { STR: 2, DEF: 2, SUP: 2 } },
      { id: "bot-2", playerId: 1, stats: { STR: 2, DEF: 2, SUP: 2 } },
      { id: "tutorial-player", playerId: 2, stats: { STR: 2, DEF: 2, SUP: 2 } }
    ];

    const currentStageData = stageResults[stageIdx];
    const currentTurn = currentStageData.turns[turnInStage - 1];
    const currentStageNum = currentStageData.stageNum;

    // Build round data with stage 1 turns and all stage turns up to current
    const roundData = {
      roundNumber: 1,
      stageNumber: currentStageNum,
      stage1Turns: [
        {
          turnNumber: 1,
          enemyIntent: round1Turn1Result.enemyIntent,
          actions: round1Turn1Result.actions,
          damageToEnemy: round1Turn1Result.damageToEnemy,
          damageToTeam: round1Turn1Result.damageToTeam,
          damageBlocked: round1Turn1Result.damageBlocked,
          healAmount: round1Turn1Result.healAmount,
          previousEnemyHealth: 10,
          previousTeamHealth: 10,
          newEnemyHealth: round1Turn1Result.enemyHealth,
          newTeamHealth: round1Turn1Result.teamHealth
        },
        {
          turnNumber: 2,
          enemyIntent: round1Turn2Result.enemyIntent,
          actions: round1Turn2Result.actions,
          damageToEnemy: round1Turn2Result.damageToEnemy,
          damageToTeam: round1Turn2Result.damageToTeam,
          damageBlocked: round1Turn2Result.damageBlocked,
          healAmount: round1Turn2Result.healAmount,
          previousEnemyHealth: round1Turn1Result.enemyHealth,
          previousTeamHealth: round1Turn1Result.teamHealth,
          newEnemyHealth: round1Turn2Result.enemyHealth,
          newTeamHealth: round1Turn2Result.teamHealth
        }
      ]
    };

    // Add turns for each stage (stage 2, 3, 4, etc.)
    for (let i = 0; i <= stageIdx; i++) {
      const stageData = stageResults[i];
      const stageKey = `stage${stageData.stageNum}Turns`;
      const turnsToShow = i < stageIdx ? stageData.turns : stageData.turns.slice(0, turnInStage);
      roundData[stageKey] = turnsToShow.map((result, idx) => ({
        turnNumber: idx + 1,
        enemyIntent: result.enemyIntent,
        actions: result.actions,
        damageToEnemy: result.damageToEnemy,
        damageToTeam: result.damageToTeam,
        damageBlocked: result.damageBlocked,
        healAmount: result.healAmount,
        previousEnemyHealth: result.previousEnemyHealth,
        previousTeamHealth: result.previousTeamHealth,
        newEnemyHealth: result.enemyHealth,
        newTeamHealth: result.teamHealth
      }));
    }

    // Build player action history showing roles for each stage
    // Convert role number to string name (e.g., 0 -> "FIGHTER") for ActionHistory component
    const playerActionHistory = stageResults.slice(0, stageIdx + 1).map((stageData) => ({
      round: 1,
      stage: stageData.stageNum,
      role: ROLE_NAMES[stageData.role]
    }));

    return {
      game: {
        enemyHealth: currentTurn.enemyHealth,
        maxEnemyHealth: 10,
        teamHealth: currentTurn.teamHealth,
        maxTeamHealth: 10,
        treatment: {
          totalPlayers: 3,
          maxRounds: 2,
          maxEnemyHealth: 10,
          maxTeamHealth: 10
        },
        get: (key) => {
          if (key === "round1Config") {
            return { maxTeamHealth: 10, maxEnemyHealth: 10 };
          }
          return undefined;
        }
      },
      player: {
        id: "tutorial-player",
        playerId: 2,
        stats: { STR: 2, DEF: 2, SUP: 2 },
        roleOrder: [ROLES.FIGHTER, ROLES.TANK, ROLES.MEDIC],
        stage: {},
        round: {
          get: (key) => {
            if (key === "playerId") return 2;
            return undefined;
          }
        },
        get: (key) => {
          if (key === "actionHistory") return playerActionHistory;
          if (key === "gamePlayerId") return 2;
          return undefined;
        }
      },
      players: players,
      round: {
        ...roundData,
        get: (key) => roundData[key]
      },
      stage: {
        name: `stage${currentStageNum}turn${turnInStage}`,
        stageType: "turn",
        stageNumber: currentStageNum,
        turnNumber: turnInStage
      }
    };
  };

  const buildAllPlayers = () => {
    if (!mockData) return [];
    return mockData.players.map((p, idx) => {
      // Player at index 2 (tutorial-player) is real in role-selection and stage-turn states
      const isPlayerReal = (currentGameState === "role-selection" || currentGameState === "stage-turn") && idx === 2;
      return {
        type: isPlayerReal ? "real" : "virtual",
        player: p,
        playerId: p.playerId,
        bot: isPlayerReal ? null : { stats: p.stats, playerId: p.playerId }
      };
    });
  };

  const handleRoleSelect = (role) => {
    setSelectedRole(role);
  };

  const handleInferredRoleChange = (playerId, roleValue) => {
    setInferredRoles(prev => ({
      ...prev,
      [playerId]: roleValue
    }));
  };

  const handleSubmit = () => {
    if (selectedRole === null) return;
    // Check that inferences are complete (both P1 and P2)
    if (inferredRoles[0] === undefined || inferredRoles[1] === undefined) return;

    // Get current stage number (stage 2 is index 0 in allStageResults)
    const currentStageNum = allStageResults.length + 2; // Stage 2, 3, 4, 5...

    // Get current health from last stage or from stage 1
    let currentEnemyHP, currentTeamHP;
    if (allStageResults.length === 0) {
      currentEnemyHP = round1Turn2Result.enemyHealth;
      currentTeamHP = round1Turn2Result.teamHealth;
    } else {
      const lastStage = allStageResults[allStageResults.length - 1];
      const lastTurn = lastStage.turns[lastStage.turns.length - 1];
      currentEnemyHP = lastTurn.enemyHealth;
      currentTeamHP = lastTurn.teamHealth;
    }

    // Simulate 2 turns for this stage
    const stageTurns = [];
    for (let turnNum = 1; turnNum <= 2 && currentEnemyHP > 0 && currentTeamHP > 0; turnNum++) {
      // Enemy always attacks in stages 2+
      const turnResult = simulateTurn(turnNum, selectedRole, currentEnemyHP, currentTeamHP, true);
      stageTurns.push(turnResult);
      currentEnemyHP = turnResult.enemyHealth;
      currentTeamHP = turnResult.teamHealth;
    }

    // Add this stage's results
    const newStageResult = {
      stageNum: currentStageNum,
      role: selectedRole,
      turns: stageTurns
    };
    const updatedStageResults = [...allStageResults, newStageResult];
    setAllStageResults(updatedStageResults);

    // Track role history
    setRoleHistory([...roleHistory, { stage: currentStageNum, role: selectedRole }]);

    // Check if game has ended
    const finalTurn = stageTurns[stageTurns.length - 1];
    const gameEnded = finalTurn.enemyHealth <= 0 || finalTurn.teamHealth <= 0;
    const maxStagesReached = currentStageNum >= 5;

    if (gameEnded || maxStagesReached) {
      // Determine outcome
      let outcomeMessage, success;
      if (finalTurn.teamHealth <= 0) {
        outcomeMessage = "Defeat! The enemy overwhelmed your team.";
        success = false;
      } else if (finalTurn.enemyHealth <= 0) {
        outcomeMessage = "Victory! Your team defeated the enemy!";
        success = true;
      } else {
        outcomeMessage = "Time's up! The battle ended in a draw.";
        success = false;
      }

      setOutcome({
        type: success ? "WIN" : "LOSE",
        message: outcomeMessage,
        success,
        enemyHealth: finalTurn.enemyHealth,
        teamHealth: finalTurn.teamHealth,
        totalTurns: 2 + updatedStageResults.reduce((sum, s) => sum + s.turns.length, 0),
        totalStages: currentStageNum
      });
    }

    // Show this stage's turn 1
    setCurrentStageIndex(updatedStageResults.length - 1);
    setCurrentTurnInStage(1);
    const newMockData = createMockDataForStage(updatedStageResults, updatedStageResults.length - 1, 1);
    setMockData(newMockData);
    setCurrentGameState("stage-turn");
    setShowDamageAnimation(true);
    setTimeout(() => setShowDamageAnimation(false), 3000);
  };

  const handleNextStageTurn = () => {
    const currentStage = allStageResults[currentStageIndex];
    const nextTurnInStage = currentTurnInStage + 1;

    // If we've shown both turns in this stage
    if (nextTurnInStage > currentStage.turns.length) {
      // Check if game has ended
      if (outcome) {
        setShowOutcome(true);
        return;
      }

      // Otherwise, go to role selection for next stage
      setSelectedRole(null);
      setInferredRoles({});
      setCurrentTurnInStage(0);
      const newMockData = createMockDataForRoleSelection();
      setMockData(newMockData);
      setCurrentGameState("role-selection");
      return;
    }

    // Show next turn in current stage
    setCurrentTurnInStage(nextTurnInStage);
    const newMockData = createMockDataForStage(allStageResults, currentStageIndex, nextTurnInStage);
    setMockData(newMockData);
    setShowDamageAnimation(true);
    setTimeout(() => setShowDamageAnimation(false), 3000);
  };

  const handlePlayAgain = () => {
    setSelectedRole(null);
    setInferredRoles({});
    setShowOutcome(false);
    setOutcome(null);
    setRound1Turn1Result(null);
    setRound1Turn2Result(null);
    setAllStageResults([]);
    setCurrentStageIndex(0);
    setCurrentTurnInStage(0);
    setRoleHistory([]);
    setIntroTutorialComplete(false);
    setRoleSelectionTutorialComplete(false);
    initializeRound1();
  };

  const handleStartMainGame = () => {
    next();
  };

  if (!mockData) return <div className="flex items-center justify-center h-screen">Loading...</div>;

  const allPlayers = buildAllPlayers();
  const isInitial = currentGameState === "initial";
  const isRound1Turn1 = currentGameState === "round1-turn1";
  const isRound1Turn2 = currentGameState === "round1-turn2";
  const isRoleSelection = currentGameState === "role-selection";
  const isStageTurn = currentGameState === "stage-turn";

  // Get current stage number for display
  const getCurrentStageNum = () => {
    if (isInitial || isRound1Turn1 || isRound1Turn2) return 1;
    if (isRoleSelection) return allStageResults.length + 2; // Next stage to play
    if (isStageTurn && allStageResults.length > 0) {
      return allStageResults[currentStageIndex].stageNum;
    }
    return 1;
  };

  // Get current turn result for display
  let currentTurnResult = null;
  if (isRound1Turn1 && round1Turn1Result) {
    currentTurnResult = round1Turn1Result;
  } else if (isRound1Turn2 && round1Turn2Result) {
    currentTurnResult = round1Turn2Result;
  } else if (isStageTurn && allStageResults.length > 0 && currentTurnInStage > 0) {
    const currentStage = allStageResults[currentStageIndex];
    currentTurnResult = currentStage.turns[currentTurnInStage - 1];
  }

  const content = (
    <MockDataProvider mockData={mockData}>
      <div className="fixed inset-0 bg-gradient-to-b from-blue-400 to-blue-600 flex items-center justify-center p-2" data-tutorial-id="full-screen">
        <div className="w-full h-full flex items-center justify-center" style={{ maxWidth: '1400px' }}>
          {/* Battle Screen */}
          <div className="bg-white rounded-lg shadow-2xl border-4 border-gray-800 w-full h-full flex overflow-hidden relative">
              {/* Left Column - Game Interface */}
              <div className="flex-1 flex flex-col min-w-0">
                {/* Stage Header */}
                <div className="bg-gray-800 text-white text-center flex-shrink-0 rounded-tl-lg flex items-center justify-center" style={{ height: '40px' }}>
                  <h1 className="text-lg font-bold">
                    Tutorial 2 - Stage {getCurrentStageNum()}{isStageTurn ? ` (Turn ${currentTurnInStage})` : ""}
                  </h1>
                </div>

                {/* Battle Field */}
                <div className="flex-shrink-0" style={{ height: '35vh', minHeight: '250px', maxHeight: '400px' }}>
                  <BattleField
                    enemyHealth={mockData.game.enemyHealth}
                    maxEnemyHealth={mockData.game.maxEnemyHealth}
                    teamHealth={mockData.game.teamHealth}
                    maxHealth={mockData.game.maxTeamHealth}
                    enemyIntent={currentTurnResult?.enemyIntent || null}
                    isRevealStage={currentTurnResult !== null}
                    showDamageAnimation={showDamageAnimation}
                    damageToEnemy={currentTurnResult?.damageToEnemy || 0}
                    damageToTeam={currentTurnResult?.damageToTeam || 0}
                    healAmount={currentTurnResult?.healAmount || 0}
                    actions={currentTurnResult?.actions || []}
                    allPlayers={allPlayers}
                    currentPlayerGameId={2}
                    previousEnemyHealth={currentTurnResult?.previousEnemyHealth || mockData.game.enemyHealth}
                    previousTeamHealth={currentTurnResult?.previousTeamHealth || mockData.game.teamHealth}
                    bossDamage={6}
                    enemyAttackProbability={0.5}
                  />
                </div>

                {/* Role Selection or Turn Results */}
                <div className="bg-white border-t-4 border-gray-700 flex-1 min-h-0 flex flex-col">
                  <div className="flex-1 p-4 flex items-center justify-center overflow-auto">
                    {/* Stage 1 Turn 1 Results */}
                    {isRound1Turn1 && round1Turn1Result && (
                      <div className="w-full">
                        <ResultsPanel
                          stageNumber={1}
                          turnNumber={1}
                          actions={round1Turn1Result.actions}
                          allPlayers={allPlayers}
                          currentPlayerGameId={2}
                          enemyIntent={round1Turn1Result.enemyIntent}
                          previousTeamHealth={10}
                          newTeamHealth={round1Turn1Result.teamHealth}
                          previousEnemyHealth={10}
                          newEnemyHealth={round1Turn1Result.enemyHealth}
                          damageToTeam={round1Turn1Result.damageToTeam}
                          damageToEnemy={round1Turn1Result.damageToEnemy}
                          damageBlocked={round1Turn1Result.damageBlocked}
                          healAmount={round1Turn1Result.healAmount}
                          onNextTurn={showRound1Turn2}
                          nextButtonLabel="Continue to Turn 2"
                        />
                      </div>
                    )}

                    {/* Stage 1 Turn 2 Results */}
                    {isRound1Turn2 && round1Turn2Result && (
                      <div className="w-full">
                        <ResultsPanel
                          stageNumber={1}
                          turnNumber={2}
                          actions={round1Turn2Result.actions}
                          allPlayers={allPlayers}
                          currentPlayerGameId={2}
                          enemyIntent={round1Turn2Result.enemyIntent}
                          previousTeamHealth={round1Turn1Result.teamHealth}
                          newTeamHealth={round1Turn2Result.teamHealth}
                          previousEnemyHealth={round1Turn1Result.enemyHealth}
                          newEnemyHealth={round1Turn2Result.enemyHealth}
                          damageToTeam={round1Turn2Result.damageToTeam}
                          damageToEnemy={round1Turn2Result.damageToEnemy}
                          damageBlocked={round1Turn2Result.damageBlocked}
                          healAmount={round1Turn2Result.healAmount}
                          onNextTurn={completeRound1}
                          nextButtonLabel="Proceed to Role Selection"
                        />
                      </div>
                    )}

                    {/* Role Selection */}
                    {isRoleSelection && (
                      <div className="w-full max-w-4xl">
                        <div data-tutorial-id="action-menu">
                          <ActionMenu
                            selectedRole={selectedRole}
                            onRoleSelect={handleRoleSelect}
                            onSubmit={handleSubmit}
                            isRoleCommitted={false}
                            currentRole={null}
                            roundsRemaining={0}
                            submitted={false}
                            roleOrder={[ROLES.FIGHTER, ROLES.TANK, ROLES.MEDIC]}
                            otherPlayersStatus={[
                              { odId: "bot-1", playerId: 0, isBot: true, submitted: true },
                              { odId: "bot-2", playerId: 1, isBot: true, submitted: true }
                            ]}
                            inferredRoles={inferredRoles}
                            onInferredRoleChange={handleInferredRoleChange}
                            showInference={true}
                          />
                        </div>
                      </div>
                    )}

                    {/* Stage Turn Results */}
                    {isStageTurn && allStageResults.length > 0 && currentTurnInStage > 0 && (() => {
                      const currentStage = allStageResults[currentStageIndex];
                      const currentTurn = currentStage.turns[currentTurnInStage - 1];
                      const nextButtonLabel = currentTurnInStage < currentStage.turns.length
                        ? `Continue to Turn ${currentTurnInStage + 1}`
                        : outcome
                          ? "See Results"
                          : `Choose Role for Stage ${currentStage.stageNum + 1}`;
                      return (
                        <div className="w-full">
                          <ResultsPanel
                            stageNumber={currentStage.stageNum}
                            turnNumber={currentTurnInStage}
                            actions={currentTurn.actions}
                            allPlayers={allPlayers}
                            currentPlayerGameId={2}
                            enemyIntent={currentTurn.enemyIntent}
                            previousTeamHealth={currentTurn.previousTeamHealth}
                            newTeamHealth={currentTurn.teamHealth}
                            previousEnemyHealth={currentTurn.previousEnemyHealth}
                            newEnemyHealth={currentTurn.enemyHealth}
                            damageToTeam={currentTurn.damageToTeam}
                            damageToEnemy={currentTurn.damageToEnemy}
                            damageBlocked={currentTurn.damageBlocked}
                            healAmount={currentTurn.healAmount}
                            onNextTurn={handleNextStageTurn}
                            nextButtonLabel={nextButtonLabel}
                          />
                        </div>
                      );
                    })()}
                  </div>
                </div>
              </div>

              {/* Right Column - Battle History */}
              <div className="bg-gray-50 border-l-4 border-gray-700 overflow-hidden flex flex-col" style={{ width: '22%', minWidth: '280px', maxWidth: '350px' }}>
                <div className="bg-gray-800 text-white text-center flex-shrink-0 rounded-tr-lg flex items-center justify-center" style={{ height: '40px' }}>
                  <h3 className="text-sm font-bold">📜 Battle History</h3>
                </div>
                <div className="flex-1 overflow-auto p-3 bg-white">
                  <ActionHistory />
                </div>
              </div>

              {/* Outcome Overlay - covers left panel only, leaves history visible */}
              {showOutcome && outcome && (
                <div className="absolute top-0 bottom-0 left-0 bg-black bg-opacity-60 flex items-center justify-center z-50" style={{ right: 'calc(22% + 4px)', minWidth: 'calc(100% - 354px)' }}>
                  <div className={`${outcome.success ? 'bg-green-50 border-green-400' : 'bg-red-50 border-red-400'} border-4 rounded-xl p-8 max-w-2xl w-full shadow-2xl mx-4`}>
                    {/* Icon and Title */}
                    <div className="text-center mb-6">
                      <div className="text-8xl mb-4">{outcome.success ? '🎉' : '💀'}</div>
                      <h1 className={`text-5xl font-bold ${outcome.success ? 'text-green-700' : 'text-red-700'} mb-2`}>
                        {outcome.success ? 'Victory!' : 'Defeat'}
                      </h1>
                      <p className="text-xl text-gray-700">{outcome.message}</p>
                    </div>

                    {/* Explanation based on role choice */}
                    <div className="bg-white rounded-lg p-6 mb-6 border-2 border-gray-300">
                      <h3 className="text-lg font-bold text-gray-800 mb-4 text-center">What Happened</h3>

                      {selectedRole === ROLES.FIGHTER && (
                        <div className="text-gray-700 space-y-3">
                          <p>
                            <span className="font-semibold">You chose Fighter 🤺</span> — Your attacks added to the team's damage output.
                          </p>
                          <p className={`font-semibold ${outcome.success ? 'text-green-600' : 'text-red-600'}`}>
                            {outcome.success
                              ? "Your additional damage helped defeat the enemy!"
                              : "The team needed more protection to survive."}
                          </p>
                        </div>
                      )}

                      {selectedRole === ROLES.TANK && (
                        <div className="text-gray-700 space-y-3">
                          <p>
                            <span className="font-semibold">You primarily chose Tank 🛡️</span> — You blocked to reduce incoming damage.
                          </p>
                          <p>
                            <span className="font-semibold text-amber-600">Remember:</span> Blocking doesn't stack — only the highest DEF applies. With a Tank already on the team, your block didn't add extra protection.
                          </p>
                          <p className={`font-semibold ${outcome.success ? 'text-green-600' : 'text-red-600'}`}>
                            {outcome.success
                              ? "The team managed to win despite the redundant defense."
                              : "The team lacked damage output to defeat the enemy in time."}
                          </p>
                        </div>
                      )}

                      {selectedRole === ROLES.MEDIC && (
                        <div className="text-gray-700 space-y-3">
                          <p>
                            <span className="font-semibold">You chose Medic 💚</span> — You healed the team when damaged.
                          </p>
                          <p>
                            <span className="font-semibold text-green-600">Good news:</span> Healing does stack — multiple Medics can heal together for greater effect.
                          </p>
                          <p className={`font-semibold ${outcome.success ? 'text-green-600' : 'text-red-600'}`}>
                            {outcome.success
                              ? "Your healing kept the team alive to secure victory!"
                              : "The team needed more damage output to defeat the enemy."}
                          </p>
                        </div>
                      )}
                    </div>

                    {/* Final Stats */}
                    <div className="bg-gray-100 rounded-lg p-4 mb-6 border border-gray-300">
                      <div className="text-center text-sm text-gray-600 mb-3">
                        Battle ended after {outcome.totalTurns || (2 + allStageResults.reduce((sum, s) => sum + s.turns.length, 0))} turns ({outcome.totalStages || (allStageResults.length + 1)} stages)
                      </div>
                      <div className="grid grid-cols-2 gap-4 mb-4">
                        <div className="text-center">
                          <div className="text-sm text-gray-600 mb-1">Team Health</div>
                          <div className="flex items-center justify-center gap-2">
                            <div className="text-2xl">❤️</div>
                            <div className={`text-xl font-bold ${outcome.teamHealth === 0 ? 'text-gray-400 line-through' : 'text-green-600'}`}>
                              {outcome.teamHealth} / 10
                            </div>
                          </div>
                        </div>
                        <div className="text-center">
                          <div className="text-sm text-gray-600 mb-1">Enemy Health</div>
                          <div className="flex items-center justify-center gap-2">
                            <div className="text-2xl">👹</div>
                            <div className={`text-xl font-bold ${outcome.enemyHealth === 0 ? 'text-gray-400 line-through' : 'text-red-600'}`}>
                              {outcome.enemyHealth} / 10
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Reveal actual roles */}
                      <div className="pt-4 border-t border-gray-300">
                        <p className="text-sm text-gray-700 mb-3 text-center font-semibold">
                          Teammate Roles (in the real game, you'll need to infer these from action patterns):
                        </p>
                        <div className="flex gap-3 justify-center mb-4">
                          <div className="text-center bg-gray-100 rounded p-2">
                            <div className="text-2xl mb-1">{ROLE_ICONS.TANK}</div>
                            <div className="text-xs text-gray-600">P1: Tank</div>
                            <div className="text-xs text-gray-500">(Blocks when attacked)</div>
                          </div>
                          <div className="text-center bg-gray-100 rounded p-2">
                            <div className="text-2xl mb-1">{ROLE_ICONS.MEDIC}</div>
                            <div className="text-xs text-gray-600">P2: Medic</div>
                            <div className="text-xs text-gray-500">(Heals when damaged)</div>
                          </div>
                        </div>
                        {roleHistory.length > 0 && (
                          <div className="mt-3">
                            <p className="text-sm text-gray-700 mb-2 text-center font-semibold">Your Role Choices:</p>
                            <div className="flex gap-2 justify-center flex-wrap">
                              {roleHistory.map((rh, idx) => (
                                <div key={idx} className="text-center bg-blue-100 border-2 border-blue-400 rounded p-2">
                                  <div className="text-xl mb-1">{ROLE_ICONS[ROLE_NAMES[rh.role]]}</div>
                                  <div className="text-xs text-gray-600">Stage {rh.stage}</div>
                                  <div className="text-xs text-gray-500">{ROLE_LABELS[rh.role]}</div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Action Buttons */}
                    <div className="flex justify-center gap-4">
                      <button
                        onClick={handlePlayAgain}
                        className="bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 px-6 rounded-lg text-lg shadow-lg transition-colors"
                      >
                        {outcome.success ? 'Play Again' : 'Try Again'}
                      </button>
                      {outcome.success && (
                        <button
                          onClick={handleStartMainGame}
                          className="bg-green-600 hover:bg-green-700 text-white font-bold py-3 px-6 rounded-lg text-lg shadow-lg transition-colors"
                        >
                          Start Main Game →
                        </button>
                      )}
                    </div>
                  </div>
                </div>
              )}
          </div>
        </div>
      </div>
    </MockDataProvider>
  );

  // Show intro tutorial at the start (empty initial screen)
  if (isInitial && !introTutorialComplete) {
    return (
      <TutorialWrapper steps={introTutorialSteps} onComplete={handleIntroTutorialComplete}>
        {content}
      </TutorialWrapper>
    );
  }

  // Show role selection tutorial when reaching that stage
  if (isRoleSelection && !roleSelectionTutorialComplete) {
    return (
      <TutorialWrapper steps={roleSelectionTutorialSteps} onComplete={handleRoleSelectionTutorialComplete}>
        {content}
      </TutorialWrapper>
    );
  }

  return content;
}

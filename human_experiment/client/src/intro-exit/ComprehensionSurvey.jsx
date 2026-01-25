import React, { useState } from "react";

const questions = [
  {
    id: 1,
    question: "Suppose the enemy intends to attack for 5 damage. Both P1 (3 DEF) and P2 (1 DEF) play the Block action, whereas P3 plays Attack. How much damage does the enemy end up doing?",
    options: [
      { label: "A", text: "1 damage (4 = 3 + 1 damage blocked)" },
      { label: "B", text: "2 damage (3 = highest-of(3, 1) damage blocked)" },
      { label: "C", text: "4 damage (1 = lowest-of(3, 1) damage blocked)" },
      { label: "D", text: "No damage" }
    ],
    correctAnswer: "B",
    explanation: "Remember: when multiple players block, only the highest DEF value counts (blocking does not stack). Think about which player has the highest DEF and how much damage that would block from a 5-damage attack."
  },
  {
    id: 2,
    question: "In a new battle, the team starts with full health, and P1 chooses to be a Medic for 2 turns. The enemy attacks on turn 1, leaving the team with 8/10 health. What actions is P1 most likely to take in turns 1 and 2?",
    options: [
      { label: "A", text: "Attack, Attack" },
      { label: "B", text: "Heal, Heal" },
      { label: "C", text: "Attack, Heal" },
      { label: "D", text: "Heal, Attack" }
    ],
    correctAnswer: "C",
    explanation: "Remember: Medics heal when the team's health is not full, but attack when at full health. Consider what the team's health was at the START of each turn, not at the end."
  },
  {
    id: 3,
    question: "In a new battle, P2 chooses to be a Tank for 2 turns. The enemy attacks on turn 1 but not on turn 2. What actions is P2 most likely to take in turns 1 and 2?",
    options: [
      { label: "A", text: "Attack, Attack" },
      { label: "B", text: "Block, Block" },
      { label: "C", text: "Attack, Block" },
      { label: "D", text: "Block, Attack" }
    ],
    correctAnswer: "D",
    explanation: "Remember: Tanks block when the enemy is attacking, but attack otherwise. Think about what the enemy does on each turn and how a Tank would respond."
  },
  {
    id: 4,
    question: "Your team has 8/10 HP, composed of all (3) Medics with SUP 1. The enemy did not attack this round. What would be your team's HP at the start of the next round?",
    options: [
      { label: "A", text: "11" },
      { label: "B", text: "10" },
      { label: "C", text: "8" },
      { label: "D", text: "9" }
    ],
    correctAnswer: "B",
    explanation: "Remember: healing stacks additively (sum of all healers' SUP). Calculate the total healing from all Medics, then consider whether there's a maximum health cap."
  },
  {
    id: 5,
    question: "Your team now has 2/6 HP. The enemy has 2 STR, and attacks 100% of the time. P1 has 1 STR and 1 SUP, P2 also has 1 STR and 1 SUP, while P3 has 4 STR and 1 SUP. All players have 0 DEF. Which roles should your team play to ensure you will never die, while still dealing the maximum amount of damage?",
    options: [
      { label: "A", text: "P1: Medic, P2: Medic, P3: Medic" },
      { label: "B", text: "P1: Medic, P2: Medic, P3: Fighter" },
      { label: "C", text: "P1: Medic, P2: Fighter, P3: Fighter" },
      { label: "D", text: "P1: Medic, P2: Fighter, P3: Medic" }
    ],
    correctAnswer: "B",
    explanation: "Think about how much damage the enemy deals per turn and how much healing you need to offset it. Remember that healing stacks (sum of SUP), and consider which player would deal the most damage as a Fighter (highest STR)."
  },
  {
    id: 6,
    question: "In your team, P1 has 4 STR but 2 DEF, whereas both P2 and P3 have 2 STR but 4 DEF. All players have 0 SUP. The enemy attacks for 4 damage every turn. Which role combination should your team play to receive no damage each turn, while otherwise maximizing the damage you deal?",
    options: [
      { label: "A", text: "P1 should play Fighter. One player among P2 or P3 should play Fighter, and the other should play Tank." },
      { label: "B", text: "P3 should play Fighter. One player among P1 or P2 should play Fighter, and the other should play Tank." },
      { label: "C", text: "P1 should play Tank. One player among P2 or P3 should play Fighter, and the other should play Tank." },
      { label: "D", text: "All players should play Fighter." }
    ],
    correctAnswer: "A",
    explanation: "To block all 4 damage, you need at least 4 DEF. Check which players have enough DEF to fully block the attack. Remember that blocking doesn't stack (only the highest DEF counts), so having multiple Tanks is wasteful. Consider who should be Fighter to maximize damage output (highest STR)."
  }
];

export function ComprehensionSurvey({ next }) {
  const [answers, setAnswers] = useState({});
  const [submitted, setSubmitted] = useState(false);
  const [showFeedback, setShowFeedback] = useState(false);

  const handleAnswerChange = (questionId, answer) => {
    setAnswers(prev => ({
      ...prev,
      [questionId]: answer
    }));
    // Reset feedback when user changes an answer
    if (showFeedback) {
      setShowFeedback(false);
      setSubmitted(false);
    }
  };

  const handleSubmit = () => {
    setSubmitted(true);
    setShowFeedback(true);

    // Check if all answers are correct
    const allCorrect = questions.every(q => answers[q.id] === q.correctAnswer);
    if (allCorrect) {
      // Small delay to show success before proceeding
      setTimeout(() => {
        next();
      }, 1500);
    }
  };

  const allAnswered = questions.every(q => answers[q.id] !== undefined);
  const correctCount = questions.filter(q => answers[q.id] === q.correctAnswer).length;
  const allCorrect = correctCount === questions.length;

  return (
    <div className="fixed inset-0 bg-gradient-to-b from-blue-400 to-blue-600 flex items-center justify-center p-4 overflow-auto">
      <div className="bg-white rounded-lg shadow-2xl border-4 border-gray-800 w-full max-w-4xl my-4">
        {/* Header */}
        <div className="bg-gray-800 text-white text-center py-4 rounded-t-lg">
          <h1 className="text-2xl font-bold">Comprehension Check</h1>
          <p className="text-sm text-gray-300 mt-1">
            Please answer all questions correctly to proceed
          </p>
        </div>

        {/* Questions */}
        <div className="p-6 space-y-8 max-h-[70vh] overflow-y-auto">
          {questions.map((q, index) => {
            const isCorrect = answers[q.id] === q.correctAnswer;
            const isAnswered = answers[q.id] !== undefined;
            const showResult = showFeedback && isAnswered;

            return (
              <div
                key={q.id}
                className={`p-4 rounded-lg border-2 ${
                  showResult
                    ? isCorrect
                      ? "border-green-500 bg-green-50"
                      : "border-red-500 bg-red-50"
                    : "border-gray-200 bg-gray-50"
                }`}
              >
                <div className="flex items-start gap-3 mb-3">
                  <span className="bg-gray-800 text-white rounded-full w-7 h-7 flex items-center justify-center text-sm font-bold flex-shrink-0">
                    {index + 1}
                  </span>
                  <p className="text-gray-800 font-medium">{q.question}</p>
                </div>

                <div className="ml-10 space-y-2">
                  {q.options.map(option => {
                    const isSelected = answers[q.id] === option.label;
                    const isCorrectOption = option.label === q.correctAnswer;

                    let optionStyle = "border-gray-300 bg-white hover:border-blue-400";
                    if (showResult) {
                      if (isSelected && !isCorrectOption) {
                        // Only highlight incorrect selections in red
                        optionStyle = "border-red-500 bg-red-100";
                      } else if (isSelected && isCorrectOption) {
                        // Keep correct selections with their blue selection style
                        optionStyle = "border-blue-500 bg-blue-50";
                      }
                    } else if (isSelected) {
                      optionStyle = "border-blue-500 bg-blue-50";
                    }

                    return (
                      <label
                        key={option.label}
                        className={`flex items-center gap-3 p-3 rounded-lg border-2 cursor-pointer transition-colors ${optionStyle}`}
                      >
                        <input
                          type="radio"
                          name={`question-${q.id}`}
                          value={option.label}
                          checked={isSelected}
                          onChange={() => handleAnswerChange(q.id, option.label)}
                          className="w-4 h-4 text-blue-600"
                          disabled={showFeedback && allCorrect}
                        />
                        <span className="text-gray-700">
                          <span className="font-semibold">{option.label}:</span> {option.text}
                        </span>
                        {showResult && isSelected && !isCorrectOption && (
                          <span className="ml-auto text-red-600 font-bold">✗</span>
                        )}
                      </label>
                    );
                  })}
                </div>

                {/* Explanation for incorrect answers */}
                {showResult && !isCorrect && (
                  <div className="ml-10 mt-3 p-3 bg-yellow-50 border border-yellow-300 rounded-lg">
                    <p className="text-sm text-yellow-800">
                      <span className="font-bold">Explanation:</span> {q.explanation}
                    </p>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Footer */}
        <div className="bg-gray-100 p-4 rounded-b-lg border-t-2 border-gray-300">
          {showFeedback && (
            <div className={`text-center mb-4 p-3 rounded-lg ${
              allCorrect
                ? "bg-green-100 text-green-800"
                : "bg-red-100 text-red-800"
            }`}>
              {allCorrect ? (
                <p className="font-bold text-lg">
                  All correct! Proceeding to the game...
                </p>
              ) : (
                <p className="font-bold">
                  {correctCount}/{questions.length} correct.
                  Please review the explanations and try again.
                </p>
              )}
            </div>
          )}

          <div className="flex justify-center">
            <button
              onClick={handleSubmit}
              disabled={!allAnswered || (showFeedback && allCorrect)}
              className={`px-8 py-3 rounded-lg font-bold text-lg transition-colors ${
                !allAnswered
                  ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                  : showFeedback && allCorrect
                    ? "bg-green-500 text-white cursor-not-allowed"
                    : "bg-blue-600 hover:bg-blue-700 text-white"
              }`}
            >
              {showFeedback && allCorrect
                ? "Proceeding..."
                : showFeedback
                  ? "Try Again"
                  : "Submit Answers"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

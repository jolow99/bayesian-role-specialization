import { usePlayer, useGame } from "@empirica/core/player/classic/react";
import React, { useState } from "react";

export function ExitSurvey({ next }) {
  const labelClassName = "block text-sm font-medium text-gray-700 mb-1";
  const inputClassName =
    "appearance-none block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-blue-500 focus:border-blue-500 text-sm";
  const player = usePlayer();
  const game = useGame();

  const outcome = game.get("outcome") || "UNKNOWN";

  // Strategy and coordination questions
  const [strategy, setStrategy] = useState("");
  const [coordinationQuality, setCoordinationQuality] = useState("4");

  // Demographics
  const [age, setAge] = useState("");
  const [gender, setGender] = useState("");
  const [genderOther, setGenderOther] = useState("");
  const [nationality, setNationality] = useState("");
  const [education, setEducation] = useState("");
  const [feedback, setFeedback] = useState("");

  function handleSubmit(event) {
    event.preventDefault();
    player.set("exitSurvey", {
      // Strategy and coordination
      strategy,
      coordinationQuality,
      // Demographics
      age,
      gender: gender === "other" ? genderOther : gender,
      nationality,
      education,
      feedback,
    });
    next();
  }

  function handleEducationChange(e) {
    setEducation(e.target.value);
  }

  const outcomeConfig = {
    WIN: { emoji: "🏆", title: "Victory!", color: "green", message: "Congratulations! Your team defeated the enemy!" },
    LOSE: { emoji: "💔", title: "Defeat", color: "red", message: "Your team was defeated, but you completed the study." },
    TIMEOUT: { emoji: "⏱️", title: "Time's Up", color: "yellow", message: "The game ended. Thank you for participating!" },
  };
  const outcomeInfo = outcomeConfig[outcome] || { emoji: "📋", title: "Game Complete", color: "blue", message: "Thank you for participating!" };

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-400 to-blue-600 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-2xl border-4 border-gray-800 p-8 max-w-2xl w-full">
        {/* Outcome Banner */}
        <div className="text-center mb-6">
          <div className="text-5xl mb-2">{outcomeInfo.emoji}</div>
          <h1 className="text-2xl font-bold text-gray-800 mb-1">{outcomeInfo.title}</h1>
          <p className="text-gray-600">{outcomeInfo.message}</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          {/* Strategy and Coordination Section */}
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-gray-800 mb-4">
              Strategy and Coordination
            </h3>

            <div className="space-y-4">
              <div>
                <label className={labelClassName}>
                  How would you describe your strategy? What influenced your action choices?
                </label>
                <textarea
                  className={inputClassName}
                  dir="auto"
                  rows={3}
                  value={strategy}
                  onChange={(e) => setStrategy(e.target.value)}
                  placeholder="Please describe your strategy and decision-making process..."
                />
              </div>

              <div>
                <label className={labelClassName}>
                  How well did your team coordinate? (1 = Not at all, 7 = Very well)
                </label>
                <div className="flex gap-3 items-center justify-center mt-2">
                  {[1, 2, 3, 4, 5, 6, 7].map(val => (
                    <label key={val} className="flex flex-col items-center cursor-pointer">
                      <input
                        type="radio"
                        name="coordinationQuality"
                        value={val}
                        checked={coordinationQuality === String(val)}
                        onChange={(e) => setCoordinationQuality(e.target.value)}
                        className="sr-only"
                      />
                      <span className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold transition-colors ${
                        coordinationQuality === String(val)
                          ? "bg-blue-600 text-white"
                          : "bg-gray-200 text-gray-600 hover:bg-gray-300"
                      }`}>
                        {val}
                      </span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Demographics Section */}
          <div className="bg-gray-50 rounded-lg p-4">
            <h3 className="text-lg font-semibold text-gray-800 mb-1">
              Demographics
            </h3>
            <p className="text-sm text-gray-500 mb-4">
              All fields are optional.
            </p>

            <div className="space-y-4">
              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label htmlFor="age" className={labelClassName}>
                    Age
                  </label>
                  <input
                    id="age"
                    name="age"
                    type="number"
                    min="18"
                    max="120"
                    autoComplete="off"
                    className={inputClassName}
                    value={age}
                    onChange={(e) => setAge(e.target.value)}
                    placeholder="e.g., 25"
                  />
                </div>
                <div>
                  <label htmlFor="gender" className={labelClassName}>
                    Gender
                  </label>
                  <select
                    id="gender"
                    name="gender"
                    className={inputClassName}
                    value={gender}
                    onChange={(e) => setGender(e.target.value)}
                  >
                    <option value="">Select...</option>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                    <option value="non-binary">Non-binary</option>
                    <option value="other">Other</option>
                    <option value="prefer-not-to-say">Prefer not to say</option>
                  </select>
                </div>
                <div>
                  <label htmlFor="nationality" className={labelClassName}>
                    Nationality
                  </label>
                  <input
                    id="nationality"
                    name="nationality"
                    type="text"
                    autoComplete="off"
                    className={inputClassName}
                    value={nationality}
                    onChange={(e) => setNationality(e.target.value)}
                    placeholder="e.g., British"
                  />
                </div>
              </div>

              {gender === "other" && (
                <div>
                  <label htmlFor="genderOther" className={labelClassName}>
                    Please specify your gender
                  </label>
                  <input
                    id="genderOther"
                    name="genderOther"
                    type="text"
                    autoComplete="off"
                    className={inputClassName}
                    value={genderOther}
                    onChange={(e) => setGenderOther(e.target.value)}
                    placeholder="Enter your gender"
                  />
                </div>
              )}

              <div>
                <label className={labelClassName}>
                  Highest Education Qualification
                </label>
                <div className="grid grid-cols-2 gap-2 mt-1">
                  <Radio
                    selected={education}
                    name="education"
                    value="high-school"
                    label="High School"
                    onChange={handleEducationChange}
                  />
                  <Radio
                    selected={education}
                    name="education"
                    value="bachelor"
                    label="Bachelor's Degree"
                    onChange={handleEducationChange}
                  />
                  <Radio
                    selected={education}
                    name="education"
                    value="master"
                    label="Master's or higher"
                    onChange={handleEducationChange}
                  />
                  <Radio
                    selected={education}
                    name="education"
                    value="other"
                    label="Other"
                    onChange={handleEducationChange}
                  />
                </div>
              </div>

              <div>
                <label className={labelClassName}>
                  Any feedback or problems you encountered?
                </label>
                <textarea
                  className={inputClassName}
                  dir="auto"
                  id="feedback"
                  name="feedback"
                  rows={3}
                  value={feedback}
                  onChange={(e) => setFeedback(e.target.value)}
                  placeholder="Please share any feedback about the game..."
                />
              </div>
            </div>
          </div>

          <button
            type="submit"
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors text-lg"
          >
            Submit Survey
          </button>
        </form>
      </div>
    </div>
  );
}

export function Radio({ selected, name, value, label, onChange }) {
  const isSelected = selected === value;
  return (
    <label className={`flex items-center px-3 py-2 rounded-md cursor-pointer transition-colors text-sm ${
      isSelected
        ? "bg-blue-100 border border-blue-500 text-blue-700"
        : "bg-white border border-gray-300 text-gray-700 hover:bg-gray-50"
    }`}>
      <input
        className="sr-only"
        type="radio"
        name={name}
        value={value}
        checked={isSelected}
        onChange={onChange}
      />
      <span className={`w-4 h-4 rounded-full border-2 mr-2 flex items-center justify-center ${
        isSelected ? "border-blue-600" : "border-gray-400"
      }`}>
        {isSelected && <span className="w-2 h-2 rounded-full bg-blue-600" />}
      </span>
      {label}
    </label>
  );
}

import { usePlayer, useGame } from "@empirica/core/player/classic/react";
import React, { useState } from "react";
import { Alert } from "../components/Alert";
import { Button } from "../components/Button";

export function ExitSurvey({ next }) {
  const labelClassName = "block text-sm font-medium text-gray-700 my-2";
  const inputClassName =
    "appearance-none block w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-empirica-500 focus:border-empirica-500 sm:text-sm";
  const player = usePlayer();
  const game = useGame();

  const outcome = game.get("outcome") || "UNKNOWN";

  // Role inference questions
  const [yourRole, setYourRole] = useState("");
  const [player1Role, setPlayer1Role] = useState("");
  const [player2Role, setPlayer2Role] = useState("");
  const [player3Role, setPlayer3Role] = useState("");
  const [roleConfidence, setRoleConfidence] = useState("4");
  const [strategy, setStrategy] = useState("");
  const [coordinationQuality, setCoordinationQuality] = useState("4");

  // Demographics
  const [age, setAge] = useState("");
  const [gender, setGender] = useState("");
  const [education, setEducation] = useState("");
  const [feedback, setFeedback] = useState("");

  function handleSubmit(event) {
    event.preventDefault();
    player.set("exitSurvey", {
      // Role inference
      yourRole,
      player1Role,
      player2Role,
      player3Role,
      roleConfidence,
      strategy,
      coordinationQuality,
      // Demographics
      age,
      gender,
      education,
      feedback,
    });
    next();
  }

  function handleEducationChange(e) {
    setEducation(e.target.value);
  }

  const playerId = player.get("playerId");

  return (
    <div className="py-8 max-w-5xl mx-auto px-4 sm:px-6 lg:px-8">
      {outcome === "WIN" && (
        <Alert title="Victory!" type="success">
          <p>Congratulations! Your team defeated the enemy!</p>
        </Alert>
      )}
      {outcome === "LOSE" && (
        <Alert title="Defeat" type="error">
          <p>Your team was defeated, but you completed the study.</p>
        </Alert>
      )}
      {outcome === "TIMEOUT" && (
        <Alert title="Time's Up">
          <p>The game ended. Thank you for participating!</p>
        </Alert>
      )}

      <form
        className="mt-8 space-y-8 divide-y divide-gray-200"
        onSubmit={handleSubmit}
      >
        <div className="space-y-8 divide-y divide-gray-200">
          {/* Role Inference Section */}
          <div>
            <div>
              <h3 className="text-lg leading-6 font-medium text-gray-900">
                Role and Strategy Questions
              </h3>
              <p className="mt-1 text-sm text-gray-500">
                Please answer the following questions about your gameplay experience.
              </p>
            </div>

            <div className="space-y-6 mt-6">
              <div>
                <label className={labelClassName}>
                  What role did you think you were playing?
                </label>
                <div className="grid gap-2">
                  <Radio
                    selected={yourRole}
                    name="yourRole"
                    value="fighter"
                    label="⚔️ Fighter - Focused on attacking"
                    onChange={(e) => setYourRole(e.target.value)}
                  />
                  <Radio
                    selected={yourRole}
                    name="yourRole"
                    value="tank"
                    label="🛡️ Tank - Focused on defending"
                    onChange={(e) => setYourRole(e.target.value)}
                  />
                  <Radio
                    selected={yourRole}
                    name="yourRole"
                    value="healer"
                    label="💚 Healer - Focused on healing"
                    onChange={(e) => setYourRole(e.target.value)}
                  />
                  <Radio
                    selected={yourRole}
                    name="yourRole"
                    value="mixed"
                    label="Mixed - I didn't stick to one role"
                    onChange={(e) => setYourRole(e.target.value)}
                  />
                  <Radio
                    selected={yourRole}
                    name="yourRole"
                    value="unsure"
                    label="I'm not sure"
                    onChange={(e) => setYourRole(e.target.value)}
                  />
                </div>
              </div>

              <div className="bg-gray-50 p-4 rounded-lg border border-gray-200">
                <label className={labelClassName}>
                  What roles do you think your teammates were playing?
                </label>
                <p className="text-xs text-gray-500 mb-3">
                  Note: You were Player {playerId + 1}
                </p>
                <div className="space-y-4">
                  {[0, 1, 2].filter(pid => pid !== playerId).map((pid, idx) => (
                    <div key={pid}>
                      <label className="text-sm font-medium text-gray-600 mb-1 block">
                        Player {pid + 1}:
                      </label>
                      <select
                        className={inputClassName}
                        value={pid === 0 ? player1Role : pid === 1 ? player2Role : player3Role}
                        onChange={(e) => {
                          if (pid === 0) setPlayer1Role(e.target.value);
                          else if (pid === 1) setPlayer2Role(e.target.value);
                          else setPlayer3Role(e.target.value);
                        }}
                      >
                        <option value="">Select a role...</option>
                        <option value="fighter">⚔️ Fighter</option>
                        <option value="tank">🛡️ Tank</option>
                        <option value="healer">💚 Healer</option>
                        <option value="mixed">Mixed</option>
                        <option value="unsure">I'm not sure</option>
                      </select>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <label className={labelClassName}>
                  How confident are you in your assessment of the roles? (1 = Not at all, 7 = Very confident)
                </label>
                <div className="flex gap-2 items-center">
                  {[1, 2, 3, 4, 5, 6, 7].map(val => (
                    <label key={val} className="flex items-center">
                      <input
                        type="radio"
                        name="roleConfidence"
                        value={val}
                        checked={roleConfidence === String(val)}
                        onChange={(e) => setRoleConfidence(e.target.value)}
                        className="mr-1"
                      />
                      <span className="text-sm">{val}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div>
                <label className={labelClassName}>
                  How would you describe your strategy? What influenced your action choices?
                </label>
                <textarea
                  className={inputClassName}
                  dir="auto"
                  rows={4}
                  value={strategy}
                  onChange={(e) => setStrategy(e.target.value)}
                  placeholder="Please describe your strategy and decision-making process..."
                />
              </div>

              <div>
                <label className={labelClassName}>
                  How well did your team coordinate? (1 = Not at all, 7 = Very well)
                </label>
                <div className="flex gap-2 items-center">
                  {[1, 2, 3, 4, 5, 6, 7].map(val => (
                    <label key={val} className="flex items-center">
                      <input
                        type="radio"
                        name="coordinationQuality"
                        value={val}
                        checked={coordinationQuality === String(val)}
                        onChange={(e) => setCoordinationQuality(e.target.value)}
                        className="mr-1"
                      />
                      <span className="text-sm">{val}</span>
                    </label>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Demographics Section */}
          <div className="pt-8">
            <div>
              <h3 className="text-lg leading-6 font-medium text-gray-900">
                Demographics (Optional)
              </h3>
              <p className="mt-1 text-sm text-gray-500">
                Please provide the following information if you feel comfortable.
              </p>
            </div>

            <div className="space-y-6 mt-6">
              <div className="flex flex-row gap-4">
                <div className="flex-1">
                  <label htmlFor="age" className={labelClassName}>
                    Age
                  </label>
                  <input
                    id="age"
                    name="age"
                    type="number"
                    autoComplete="off"
                    className={inputClassName}
                    value={age}
                    onChange={(e) => setAge(e.target.value)}
                  />
                </div>
                <div className="flex-1">
                  <label htmlFor="gender" className={labelClassName}>
                    Gender
                  </label>
                  <input
                    id="gender"
                    name="gender"
                    autoComplete="off"
                    className={inputClassName}
                    value={gender}
                    onChange={(e) => setGender(e.target.value)}
                  />
                </div>
              </div>

              <div>
                <label className={labelClassName}>
                  Highest Education Qualification
                </label>
                <div className="grid gap-2">
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
                  rows={4}
                  value={feedback}
                  onChange={(e) => setFeedback(e.target.value)}
                  placeholder="Please share any feedback about the game..."
                />
              </div>

              <div className="mb-12">
                <Button type="submit">Submit</Button>
              </div>
            </div>
          </div>
        </div>
      </form>
    </div>
  );
}

export function Radio({ selected, name, value, label, onChange }) {
  return (
    <label className="text-sm font-medium text-gray-700">
      <input
        className="mr-2 shadow-sm sm:text-sm"
        type="radio"
        name={name}
        value={value}
        checked={selected === value}
        onChange={onChange}
      />
      {label}
    </label>
  );
}

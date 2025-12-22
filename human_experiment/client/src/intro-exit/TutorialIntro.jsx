import React from "react";
import { Button } from "../components/Button";

export function TutorialIntro({ next }) {
  return (
    <div className="mt-3 sm:mt-5 p-20 max-w-4xl mx-auto">
      <h3 className="text-3xl leading-6 font-bold text-gray-900 mb-6 text-center">
        Tutorial
      </h3>

      <div className="space-y-6">
        <section>
          <p className="text-sm text-gray-700 mb-3">
            Before we start the main game, you'll play through two short tutorial rounds
            to familiarize yourself with the game mechanics.
          </p>
          <p className="text-sm text-gray-700 mb-3">
            In these tutorial rounds, you'll be playing with two other teammates.
            Pay attention to how your choices affect the outcome of each round.
          </p>
        </section>

        <div className="bg-blue-50 border border-blue-200 rounded p-4">
          <p className="text-sm text-blue-800 font-semibold mb-2">
            Tutorial Goals:
          </p>
          <ul className="text-sm text-blue-800 space-y-1 list-disc list-inside">
            <li>Understand how different roles work together</li>
            <li>Learn to make strategic choices based on team composition</li>
            <li>Practice interpreting actions and outcomes</li>
          </ul>
        </div>

        <p className="text-sm text-gray-600 italic text-center">
          Remember: These are just practice rounds to help you learn!
        </p>
      </div>

      <div className="flex justify-center mt-8">
        <Button handleClick={next} autoFocus>
          <p>Start Tutorial</p>
        </Button>
      </div>
    </div>
  );
}

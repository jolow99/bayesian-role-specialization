import React, { useEffect, useState } from "react";

const COMPLETION_CODE = import.meta.env.VITE_PROLIFIC_COMPLETION_CODE || "";
const PROLIFIC_REDIRECT_URL = `https://app.prolific.com/submissions/complete?cc=${COMPLETION_CODE}`;

export function Finished() {
  const [countdown, setCountdown] = useState(5);

  useEffect(() => {
    if (!COMPLETION_CODE) return;

    const timer = setInterval(() => {
      setCountdown((prev) => {
        if (prev <= 1) {
          clearInterval(timer);
          window.location.href = PROLIFIC_REDIRECT_URL;
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => clearInterval(timer);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-b from-blue-400 to-blue-600 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-2xl border-4 border-gray-800 p-8 max-w-lg w-full">
        <div className="text-center">
          <div className="text-6xl mb-4">🎉</div>
          <h1 className="text-2xl font-bold text-gray-800 mb-4">
            Thank You for Participating!
          </h1>

          <p className="text-gray-600 mb-6">
            You have successfully completed the experiment.
          </p>

          {COMPLETION_CODE && (
            <div className="bg-green-50 border border-green-200 rounded-lg p-4 mb-6">
              <h2 className="font-semibold text-green-800 mb-2">
                Prolific Completion Code
              </h2>
              <p className="text-3xl font-mono font-bold text-green-700 tracking-wider">
                {COMPLETION_CODE}
              </p>
              <p className="text-green-600 text-sm mt-2">
                Redirecting to Prolific in {countdown} seconds...
              </p>
            </div>
          )}

          {COMPLETION_CODE && (
            <a
              href={PROLIFIC_REDIRECT_URL}
              className="inline-block bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-lg transition-colors mb-6"
            >
              Return to Prolific Now
            </a>
          )}

          <div className="bg-gray-100 rounded-lg p-4 mb-6">
            <h2 className="font-semibold text-gray-700 mb-2">
              Questions or Issues?
            </h2>
            <p className="text-gray-600 text-sm">
              If you have any questions or encounter any issues with your payment,
              please reach out to us through Prolific's messaging system.
            </p>
          </div>

          <p className="text-sm text-gray-500">
            {COMPLETION_CODE
              ? "If you are not redirected automatically, click the button above or copy the completion code."
              : "You may now close this tab."}
          </p>
        </div>
      </div>
    </div>
  );
}

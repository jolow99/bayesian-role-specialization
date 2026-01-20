import React from "react";

export function Finished() {
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

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
            <h2 className="font-semibold text-blue-800 mb-2">
              Payment Information
            </h2>
            <p className="text-blue-700 text-sm">
              Your payment will be processed through Prolific. We will be manually
              verifying submissions, so please allow some time for your payment to
              be approved.
            </p>
          </div>

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
            You may now close this tab.
          </p>
        </div>
      </div>
    </div>
  );
}

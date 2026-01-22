import { Consent as EmpiricaConsent } from "@empirica/core/player/react";
import { usePlayer } from "@empirica/core/player/classic/react";
import React from "react";

export function Consent({ onConsent }) {
  const player = usePlayer();

  const consentText = `This experiment is part of a scientific project. Your decision to participate in this experiment is entirely voluntary. There are no known or anticipated risks to participating in this experiment. There is no way for us to identify you. The only information we will have, in addition to your responses, is the timestamps of your interactions with our site. The results of our research may be presented at scientific meetings or published in scientific journals. Clicking on the "I AGREE" button indicates that you are at least 21 years of age, and agree to participate voluntarily.`;

  const handleConsent = () => {
    // Track when player consented (start of tutorial)
    if (player) {
      player.set("consentedAt", Date.now());
    }
    onConsent();
  };

  return (
    <EmpiricaConsent
      title="Do you consent to participate in this experiment?"
      text={consentText}
      buttonText="I AGREE"
      onConsent={handleConsent}
    />
  );
}

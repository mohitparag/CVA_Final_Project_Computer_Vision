/**
 * voiceCoach.js
 * Stretch Goal: Real-time accessible voice coaching using the browser's
 * built-in Web Speech API (window.speechSynthesis). 100% free & offline-capable.
 */

export class VoiceCoach {
  constructor(cooldownMs = 2600) {
    this.enabled = false;
    this.cooldownMs = cooldownMs;
    this.lastSpokenAt = 0;
    this.lastText = '';
  }

  setEnabled(flag) {
    this.enabled = Boolean(flag);
    if (!this.enabled && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
  }

  speak(text, priority = false) {
    if (!this.enabled || !text || !('speechSynthesis' in window)) return;

    const now = performance.now();
    if (!priority && text === this.lastText && now - this.lastSpokenAt < this.cooldownMs * 1.6) {
      return;
    }
    if (!priority && now - this.lastSpokenAt < this.cooldownMs) {
      return;
    }

    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    utter.rate = 1.05;
    utter.pitch = 1.0;
    utter.volume = 0.9;
    window.speechSynthesis.speak(utter);

    this.lastSpokenAt = now;
    this.lastText = text;
  }
}

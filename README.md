# SmartCycle
AI-powered recycling contamination detection and disposal guidance system.

## Mobile app (`mobile/`)

React Native + Expo app. Flow: **capture photo → analyze material & contamination → preparation advice → re-scan to verify readiness.**

> Analysis is currently **simulated** in `mobile/src/analyze.ts`. Before each scan, the tester picks a demo scenario (dirty, clean, still dirty after re-scan, or uncertain), and the result is labelled "Demo — simulated analysis". The photo itself is not analyzed. Recycling guidance is **illustrative** until we connect verified local rules. A real vision model will be wired in later through a backend.

### Run it on an iPhone (from Windows, no Mac needed)

1. Install **Expo Go** from the App Store on your iPhone.
2. Install [Node.js LTS](https://nodejs.org/) on your PC.
3. In a terminal:
   ```bash
   cd mobile
   npm install
   npx expo start
   ```
4. Put the phone and PC on the **same Wi-Fi**, then scan the QR code in the terminal with the iPhone **Camera** app. It opens in Expo Go.
   - Can't connect (school or firewalled network)? Use `npx expo start --tunnel`.
5. Edit code and save. The app reloads on the phone automatically.

### Useful commands (inside `mobile/`)

```bash
npx expo install <package>  # add a library (picks versions compatible with our Expo SDK)
npx tsc --noEmit            # typecheck
npx expo lint               # lint
npx expo-doctor             # diagnose dependency/config problems
```

### Code layout

```
mobile/
  App.tsx                 # flow state machine: capture → analyzing → result → verified
  src/
    analyze.ts            # analyzeImage() + demo scenarios — backend call later
    takePhoto.ts          # camera permission, capture, cancel/denied handling (expo-image-picker)
    components/           # DemoBanner, IllustrativeGuidanceNotice, ScenarioPicker
    types.ts              # AnalysisResult shape
    theme.ts              # colors + shared styles
    screens/
      CaptureScreen.tsx   # intro + "Scan an item"
      ResultScreen.tsx    # not ready (contamination + prep steps) or uncertain (retake photo)
      VerifyScreen.tsx    # "Ready to recycle ✅"
```

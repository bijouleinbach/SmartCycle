import { useRef, useState } from 'react';
import { ActivityIndicator, Alert, Pressable, StyleSheet, Text, View } from 'react-native';
import { StatusBar } from 'expo-status-bar';
import { SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import { analyzeImage } from './src/analyze';
import { takePhoto } from './src/takePhoto';
import { AnalysisResult, DemoScenario } from './src/types';
import { CaptureScreen } from './src/screens/CaptureScreen';
import { ResultScreen } from './src/screens/ResultScreen';
import { VerifyScreen } from './src/screens/VerifyScreen';
import { colors, shared } from './src/theme';

// The whole flow is a small state machine:
// capture → analyzing → result (not_ready / uncertain) → re-scan → analyzing → …
//                                                                → verified (ready)
// Readiness comes only from the analysis result, never from the act of re-scanning.
type Step =
  | { name: 'capture' }
  | { name: 'analyzing'; photoUri: string }
  | { name: 'result'; photoUri: string; result: AnalysisResult }
  | { name: 'verified'; photoUri: string; result: AnalysisResult };

export default function App() {
  const [step, setStep] = useState<Step>({ name: 'capture' });
  const [scenario, setScenario] = useState<DemoScenario>('dirty');
  const [notice, setNotice] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false); // blocks double taps before React re-renders

  async function scan() {
    if (busyRef.current) return;
    busyRef.current = true;
    setBusy(true);
    setNotice(null);
    const previousStep = step;

    try {
      const photo = await takePhoto();
      if (photo.kind === 'cancelled') {
        setNotice('Scan cancelled — no photo was taken.');
        return;
      }
      if (photo.kind !== 'photo') return; // denied / error already showed an alert

      setStep({ name: 'analyzing', photoUri: photo.uri });
      try {
        const result = await analyzeImage(photo.uri, { scenario });
        setStep(
          result.status === 'ready'
            ? { name: 'verified', photoUri: photo.uri, result }
            : { name: 'result', photoUri: photo.uri, result },
        );
      } catch {
        Alert.alert('Analysis failed', 'Please try scanning again.');
        setStep(previousStep);
      }
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }

  function startOver() {
    setNotice(null);
    setStep({ name: 'capture' });
  }

  return (
    <SafeAreaProvider>
      <SafeAreaView style={shared.screen}>
        {notice && (
          <Pressable style={styles.notice} onPress={() => setNotice(null)}>
            <Text style={styles.noticeText}>{notice}</Text>
            <Text style={styles.noticeDismiss}>✕</Text>
          </Pressable>
        )}

        {step.name === 'capture' && (
          <CaptureScreen
            scenario={scenario}
            onScenarioChange={setScenario}
            onScan={scan}
            busy={busy}
          />
        )}

        {step.name === 'analyzing' && (
          <View style={styles.center}>
            <ActivityIndicator size="large" color={colors.green} />
            <Text style={shared.body}>Analyzing material and contamination…</Text>
          </View>
        )}

        {step.name === 'result' && (
          <ResultScreen
            photoUri={step.photoUri}
            result={step.result}
            scenario={scenario}
            onScenarioChange={setScenario}
            onRescan={scan}
            onStartOver={startOver}
            busy={busy}
          />
        )}

        {step.name === 'verified' && (
          <VerifyScreen photoUri={step.photoUri} result={step.result} onScanAnother={startOver} />
        )}
      </SafeAreaView>
      <StatusBar style="dark" />
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  center: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 16, padding: 20 },
  notice: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginHorizontal: 20,
    marginTop: 8,
    padding: 12,
    borderRadius: 10,
    backgroundColor: '#F1F3F4',
  },
  noticeText: { flex: 1, fontSize: 14, color: colors.text },
  noticeDismiss: { fontSize: 14, color: colors.muted },
});

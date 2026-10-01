import { Image, Pressable, ScrollView, Text, View } from 'react-native';
import { DemoBanner, IllustrativeGuidanceNotice } from '../components/Notices';
import { ScenarioPicker } from '../components/ScenarioPicker';
import { AnalysisResult, DemoScenario } from '../types';
import { colors, shared } from '../theme';

interface Props {
  photoUri: string;
  /** status is 'not_ready' or 'uncertain' */
  result: AnalysisResult;
  scenario: DemoScenario;
  onScenarioChange: (scenario: DemoScenario) => void;
  onRescan: () => void;
  onStartOver: () => void;
  busy: boolean;
}

export function ResultScreen({
  photoUri,
  result,
  scenario,
  onScenarioChange,
  onRescan,
  onStartOver,
  busy,
}: Props) {
  const uncertain = result.status === 'uncertain';

  return (
    <ScrollView style={shared.screen} contentContainerStyle={shared.content}>
      {result.simulated && <DemoBanner />}
      <Text style={shared.title}>{uncertain ? 'Not sure yet' : 'Not ready yet'}</Text>
      <Image source={{ uri: photoUri }} style={shared.photo} />

      <View style={{ gap: 4 }}>
        <Text style={shared.heading}>{result.itemLabel}</Text>
        <Text style={shared.muted}>
          Material: {result.material} · {Math.round(result.confidence * 100)}% confident
        </Text>
      </View>

      {uncertain ? (
        <View style={{ gap: 6, padding: 16, borderRadius: 12, backgroundColor: colors.blueLight }}>
          <Text style={[shared.heading, { color: colors.blue }]}>📷 Please take another photo</Text>
          <Text style={shared.body}>
            We couldn’t tell the material or whether it’s clean. Don’t recycle it based on this
            result.
          </Text>
          {result.advice.map((tip) => (
            <Text key={tip} style={shared.body}>• {tip}</Text>
          ))}
        </View>
      ) : (
        <>
          <View style={{ gap: 6, padding: 16, borderRadius: 12, backgroundColor: colors.amberLight }}>
            <Text style={[shared.heading, { color: colors.amber }]}>⚠️ Contamination found</Text>
            {result.contaminants.map((c) => (
              <Text key={c} style={shared.body}>• {c}</Text>
            ))}
          </View>

          <View style={{ gap: 6 }}>
            <Text style={shared.heading}>How to prepare it</Text>
            {result.advice.map((step, i) => (
              <Text key={step} style={shared.body}>
                {i + 1}. {step}
              </Text>
            ))}
          </View>
          {!result.guidanceVerified && <IllustrativeGuidanceNotice />}
        </>
      )}

      {result.simulated && (
        <ScenarioPicker value={scenario} onChange={onScenarioChange} disabled={busy} />
      )}

      <Pressable
        style={[shared.button, busy && shared.disabled]}
        onPress={onRescan}
        disabled={busy}
      >
        <Text style={shared.buttonText}>
          {uncertain ? 'Take another photo' : 'Done — re-scan to verify'}
        </Text>
      </Pressable>
      <Pressable
        style={[shared.secondaryButton, busy && shared.disabled]}
        onPress={onStartOver}
        disabled={busy}
      >
        <Text style={shared.secondaryButtonText}>Start over</Text>
      </Pressable>
    </ScrollView>
  );
}

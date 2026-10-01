import { Pressable, ScrollView, Text, View } from 'react-native';
import { ScenarioPicker } from '../components/ScenarioPicker';
import { DemoScenario } from '../types';
import { colors, shared } from '../theme';

interface Props {
  scenario: DemoScenario;
  onScenarioChange: (scenario: DemoScenario) => void;
  onScan: () => void;
  busy: boolean;
}

export function CaptureScreen({ scenario, onScenarioChange, onScan, busy }: Props) {
  return (
    <ScrollView style={shared.screen} contentContainerStyle={shared.content}>
      <Text style={shared.title}>♻️ SmartCycle</Text>
      <Text style={shared.body}>
        Snap a photo of an item before you recycle it. We’ll identify the material, spot
        contamination, and tell you how to prep it.
      </Text>

      <View style={{ gap: 8, padding: 16, borderRadius: 12, backgroundColor: colors.greenLight }}>
        <Text style={shared.heading}>How it works</Text>
        <Text style={shared.body}>1. Scan the item</Text>
        <Text style={shared.body}>2. See material + contamination</Text>
        <Text style={shared.body}>3. Follow the prep steps</Text>
        <Text style={shared.body}>4. Re-scan to confirm it’s ready</Text>
      </View>

      <ScenarioPicker value={scenario} onChange={onScenarioChange} disabled={busy} />

      <Pressable
        style={[shared.button, busy && shared.disabled]}
        onPress={onScan}
        disabled={busy}
      >
        <Text style={shared.buttonText}>📷  Scan an item</Text>
      </Pressable>
    </ScrollView>
  );
}

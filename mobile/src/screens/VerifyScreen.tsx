import { Image, Pressable, ScrollView, Text, View } from 'react-native';
import { DemoBanner, IllustrativeGuidanceNotice } from '../components/Notices';
import { AnalysisResult } from '../types';
import { colors, shared } from '../theme';

interface Props {
  photoUri: string;
  /** status is 'ready' */
  result: AnalysisResult;
  onScanAnother: () => void;
}

export function VerifyScreen({ photoUri, result, onScanAnother }: Props) {
  return (
    <ScrollView style={shared.screen} contentContainerStyle={shared.content}>
      {result.simulated && <DemoBanner />}
      <Text style={shared.title}>Ready to recycle ✅</Text>
      <Image source={{ uri: photoUri }} style={shared.photo} />

      <View style={{ gap: 6, padding: 16, borderRadius: 12, backgroundColor: colors.greenLight }}>
        <Text style={[shared.heading, { color: colors.green }]}>{result.itemLabel}</Text>
        <Text style={shared.body}>No contamination detected. Put it in the recycling bin.</Text>
        <Text style={shared.muted}>
          Material: {result.material} · {Math.round(result.confidence * 100)}% confident
        </Text>
      </View>
      {!result.guidanceVerified && <IllustrativeGuidanceNotice />}

      <Pressable style={shared.button} onPress={onScanAnother}>
        <Text style={shared.buttonText}>Scan another item</Text>
      </Pressable>
    </ScrollView>
  );
}

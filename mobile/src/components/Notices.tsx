import { StyleSheet, Text, View } from 'react-native';
import { colors } from '../theme';

/** Shown on every result while analysis is simulated. */
export function DemoBanner() {
  return (
    <View style={[styles.box, { backgroundColor: colors.demoLight, borderColor: colors.demo }]}>
      <Text style={[styles.title, { color: colors.demo }]}>Demo — simulated analysis</Text>
      <Text style={styles.body}>
        This result comes from the demo scenario you picked. The photo was not analyzed.
      </Text>
    </View>
  );
}

/** Shown wherever we give recycling guidance until local rules are verified. */
export function IllustrativeGuidanceNotice() {
  return (
    <View style={[styles.box, { backgroundColor: '#F5F5F5', borderColor: colors.border }]}>
      <Text style={styles.title}>Illustrative guidance</Text>
      <Text style={styles.body}>
        Not yet checked against your local recycling rules. Rules vary by city, so confirm
        with your local recycling program.
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  box: { borderWidth: 1, borderRadius: 10, padding: 12, gap: 2 },
  title: { fontSize: 14, fontWeight: '700', color: colors.text },
  body: { fontSize: 13, color: colors.muted, lineHeight: 18 },
});

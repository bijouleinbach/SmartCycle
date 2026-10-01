import { Pressable, StyleSheet, Text, View } from 'react-native';
import { DEMO_SCENARIOS } from '../analyze';
import { DemoScenario } from '../types';
import { colors, shared } from '../theme';

interface Props {
  value: DemoScenario;
  onChange: (scenario: DemoScenario) => void;
  disabled?: boolean;
}

export function ScenarioPicker({ value, onChange, disabled }: Props) {
  return (
    <View style={styles.container} accessibilityRole="radiogroup">
      <Text style={[shared.heading, { color: colors.demo }]}>Demo scenario for next scan</Text>
      {DEMO_SCENARIOS.map(({ id, label }) => {
        const selected = id === value;
        return (
          <Pressable
            key={id}
            onPress={() => onChange(id)}
            disabled={disabled}
            accessibilityRole="radio"
            accessibilityState={{ checked: selected, disabled }}
            style={[styles.row, selected && styles.rowSelected]}
          >
            <View style={[styles.dot, selected && styles.dotSelected]} />
            <Text style={shared.body}>{label}</Text>
          </Pressable>
        );
      })}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    gap: 8,
    padding: 12,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: colors.demo,
    backgroundColor: colors.demoLight,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 10,
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 8,
    backgroundColor: '#fff',
    borderWidth: 1,
    borderColor: colors.border,
  },
  rowSelected: { borderColor: colors.demo },
  dot: { width: 18, height: 18, borderRadius: 9, borderWidth: 2, borderColor: colors.border },
  dotSelected: { borderColor: colors.demo, backgroundColor: colors.demo },
});

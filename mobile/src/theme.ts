import { StyleSheet } from 'react-native';

export const colors = {
  green: '#2E7D32',
  greenLight: '#E8F5E9',
  amber: '#B26A00',
  amberLight: '#FFF4E0',
  blue: '#1E5AA8',
  blueLight: '#E6EFFA',
  demo: '#6A1B9A',
  demoLight: '#F3E8FA',
  border: '#DADCE0',
  text: '#1B1B1B',
  muted: '#5F6368',
  bg: '#FFFFFF',
};

export const shared = StyleSheet.create({
  screen: { flex: 1, backgroundColor: colors.bg },
  content: { padding: 20, gap: 16 },
  title: { fontSize: 28, fontWeight: '700', color: colors.text },
  heading: { fontSize: 18, fontWeight: '600', color: colors.text },
  body: { fontSize: 16, color: colors.text, lineHeight: 22 },
  muted: { fontSize: 14, color: colors.muted },
  photo: { width: '100%', aspectRatio: 1, borderRadius: 12, backgroundColor: '#eee' },
  button: {
    backgroundColor: colors.green,
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  buttonText: { color: '#fff', fontSize: 17, fontWeight: '600' },
  disabled: { opacity: 0.5 },
  secondaryButton: {
    paddingVertical: 14,
    borderRadius: 12,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.green,
  },
  secondaryButtonText: { color: colors.green, fontSize: 16, fontWeight: '600' },
});

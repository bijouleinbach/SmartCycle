import * as ImagePicker from 'expo-image-picker';
import { Alert, Linking } from 'react-native';

export type PhotoOutcome =
  | { kind: 'photo'; uri: string }
  | { kind: 'cancelled' }
  | { kind: 'denied' }
  | { kind: 'error' };

/**
 * Asks for camera permission, opens the camera, and reports what happened.
 * Shows its own alert for denied permission and camera errors.
 */
export async function takePhoto(): Promise<PhotoOutcome> {
  try {
    const permission = await ImagePicker.requestCameraPermissionsAsync();
    if (!permission.granted) {
      showPermissionAlert(permission.canAskAgain);
      return { kind: 'denied' };
    }

    const result = await ImagePicker.launchCameraAsync({
      mediaTypes: ['images'],
      quality: 0.6, // smaller upload once we call a real backend
    });

    if (result.canceled || result.assets.length === 0) return { kind: 'cancelled' };
    return { kind: 'photo', uri: result.assets[0].uri };
  } catch {
    // e.g. no camera available (iOS Simulator)
    Alert.alert('Camera unavailable', 'Could not open the camera on this device.');
    return { kind: 'error' };
  }
}

function showPermissionAlert(canAskAgain: boolean) {
  if (canAskAgain) {
    Alert.alert('Camera access needed', 'SmartCycle needs the camera to scan items.');
    return;
  }
  // iOS only shows the system prompt once; after that the user must use Settings.
  // In Expo Go, the permission belongs to the Expo Go app.
  Alert.alert(
    'Camera access is off',
    'Turn on camera access in Settings to scan items.',
    [
      { text: 'Not now', style: 'cancel' },
      { text: 'Open Settings', onPress: () => Linking.openSettings() },
    ],
  );
}

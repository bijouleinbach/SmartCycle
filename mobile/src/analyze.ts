import { AnalysisResult, DemoScenario } from './types';

export const DEMO_SCENARIOS: { id: DemoScenario; label: string }[] = [
  { id: 'dirty', label: 'Dirty plastic container' },
  { id: 'clean', label: 'Clean plastic container' },
  { id: 'still_dirty', label: 'Still dirty after re-scan' },
  { id: 'uncertain', label: 'Uncertain — needs another photo' },
];

/**
 * Analyze a photo of an item for material and contamination.
 *
 * Phase 1: returns the simulated result for the scenario the tester picked.
 * The photo is NOT inspected, and re-scanning alone never makes an item ready.
 * Phase 2: replace the body with a call to our backend, e.g.
 *   POST {API_URL}/analyze with the image, returning an AnalysisResult.
 * Never put AI API keys in the app itself.
 */
export async function analyzeImage(
  imageUri: string,
  options: { scenario: DemoScenario },
): Promise<AnalysisResult> {
  await delay(1500); // simulate network + model latency
  return { ...SCENARIO_RESULTS[options.scenario], simulated: true, guidanceVerified: false };
}

const SCENARIO_RESULTS: Record<
  DemoScenario,
  Omit<AnalysisResult, 'simulated' | 'guidanceVerified'>
> = {
  dirty: {
    status: 'not_ready',
    material: 'plastic',
    itemLabel: 'Plastic food container (#5 PP)',
    contaminants: ['Food residue inside', 'Grease film on walls'],
    confidence: 0.9,
    advice: [
      'Scrape out leftover food into the trash or compost.',
      'Rinse with water until no food or grease is visible.',
      'Shake out excess water — it does not need to be perfectly dry.',
    ],
  },
  clean: {
    status: 'ready',
    material: 'plastic',
    itemLabel: 'Plastic food container (#5 PP)',
    contaminants: [],
    confidence: 0.93,
    advice: [],
  },
  still_dirty: {
    status: 'not_ready',
    material: 'plastic',
    itemLabel: 'Plastic food container (#5 PP)',
    contaminants: ['Food residue still visible in corners'],
    confidence: 0.88,
    advice: [
      'Use a brush or sponge on the corners and rim — a quick rinse missed some residue.',
      'Rinse again and check that no food is visible.',
      'If it cannot be cleaned, put it in the trash instead — heavily soiled items can contaminate a whole batch.',
    ],
  },
  uncertain: {
    status: 'uncertain',
    material: 'unknown',
    itemLabel: 'Unidentified item',
    contaminants: [],
    confidence: 0.38,
    advice: [
      'Move to brighter light and avoid glare.',
      'Fill the frame with the item, with nothing else in the shot.',
      'Show the inside of the container and the recycling symbol if it has one.',
    ],
  },
};

function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

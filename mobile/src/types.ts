export type Material = 'plastic' | 'glass' | 'metal' | 'paper' | 'cardboard' | 'unknown';

/**
 * - ready: no contamination found, item can go in the recycling bin
 * - not_ready: contamination found, follow the advice and re-scan
 * - uncertain: the photo wasn't clear enough to decide, take another photo
 */
export type ReadinessStatus = 'ready' | 'not_ready' | 'uncertain';

export interface AnalysisResult {
  status: ReadinessStatus;
  material: Material;
  /** Short item description, e.g. "PET plastic container". */
  itemLabel: string;
  /** Detected contaminants, e.g. "food residue". Empty when clean. */
  contaminants: string[];
  /** 0–1 confidence of the classification. */
  confidence: number;
  /** Preparation steps (not_ready) or photo tips (uncertain). Empty when ready. */
  advice: string[];
  /** True when the result came from a demo scenario, not a real model. */
  simulated: boolean;
  /** False until guidance is checked against verified local recycling rules. */
  guidanceVerified: boolean;
}

/** Demo scenarios the tester picks before each scan (Phase 1 only). */
export type DemoScenario = 'dirty' | 'clean' | 'still_dirty' | 'uncertain';

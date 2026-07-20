export type DetectionLabel = {
  id: string;
  title: string;
  points: string[];
  side: "left" | "right";
  offset: number;
};

export const detectionLabels: DetectionLabel[] = [
  {
    id: "eyes",
    title: "Eye Region",
    points: ["Blink inconsistency", "Iris deformation", "Reflection mismatch"],
    side: "left",
    offset: 0.12,
  },
  {
    id: "lips",
    title: "Lip Analysis",
    points: ["Lip-sync mismatch", "Mouth geometry"],
    side: "right",
    offset: 0.3,
  },
  {
    id: "lighting",
    title: "Lighting",
    points: ["Shadow inconsistency", "Reflection mismatch"],
    side: "left",
    offset: 0.48,
  },
  {
    id: "skin",
    title: "Skin Texture",
    points: ["GAN artifacts", "Frequency anomalies"],
    side: "right",
    offset: 0.66,
  },
  {
    id: "pose",
    title: "Head Pose",
    points: ["Landmark displacement", "Pose inconsistency"],
    side: "left",
    offset: 0.84,
  },
];

export type LandmarkGroup = {
  id: string;
  title: string;
  explanation: string;
  cx: number;
  cy: number;
};

export const landmarkGroups: LandmarkGroup[] = [
  {
    id: "eyes",
    title: "Eyes",
    explanation: "Blink rate and pupil reflection are among the hardest signals for generative models to fake consistently.",
    cx: 0.5,
    cy: 0.55,
  },
  {
    id: "eyebrows",
    title: "Eyebrows",
    explanation: "Eyebrow micro-motion rarely matches emotional context in synthetic video.",
    cx: 0.5,
    cy: 0.5,
  },
  {
    id: "nose",
    title: "Nose",
    explanation: "Nose bridge geometry anchors head-pose estimation across frames.",
    cx: 0.5,
    cy: 0.59,
  },
  {
    id: "jaw",
    title: "Jaw",
    explanation: "Jawline warping is a common artifact where the generated mask meets the original frame.",
    cx: 0.5,
    cy: 0.71,
  },
  {
    id: "lips",
    title: "Lips",
    explanation: "Phoneme-to-viseme mismatch shows up as subtle lip-sync drift under audio analysis.",
    cx: 0.5,
    cy: 0.65,
  },
  {
    id: "cheekbones",
    title: "Cheekbones",
    explanation: "Cheekbone shading inconsistency reveals mismatched light sources between face and background.",
    cx: 0.4,
    cy: 0.6,
  },
  {
    id: "boundary",
    title: "Face Boundary",
    explanation: "The blend boundary is where compression artifacts and frequency anomalies concentrate.",
    cx: 0.5,
    cy: 0.76,
  },
];

export const pipelineStages = [
  { id: "input", label: "Input Image" },
  { id: "detect", label: "Face Detection" },
  { id: "extract", label: "Feature Extraction" },
  { id: "cnn", label: "CNN / Vision Transformer" },
  { id: "forgery", label: "Forgery Analysis" },
  { id: "confidence", label: "Confidence Score" },
  { id: "prediction", label: "Prediction" },
];

export const stats = [
  { id: "accuracy", label: "Accuracy", value: 98.4, suffix: "%" },
  { id: "precision", label: "Precision", value: 97.1, suffix: "%" },
  { id: "recall", label: "Recall", value: 96.6, suffix: "%" },
  { id: "f1", label: "F1 Score", value: 96.8, suffix: "%" },
  { id: "inference", label: "Inference Time", value: 42, suffix: "ms" },
  { id: "dataset", label: "Dataset Size", value: 480, suffix: "K" },
];

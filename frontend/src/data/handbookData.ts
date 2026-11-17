import type { StudioId } from '../types';
import { HANDBOOK_CHAPTERS_PART1 } from './handbookChaptersPart1';
import { HANDBOOK_CHAPTERS_PART2 } from './handbookChaptersPart2';
import { DEEP_SUPPLEMENTS_PART_1 } from './handbookDeepPart1';
import { DEEP_SUPPLEMENTS_PART_2 } from './handbookDeepPart2';
import type { StudioChapter, StudioDeepSupplement } from './handbookTypes';

export * from './handbookTypes';

const ALL_SUPPLEMENTS: Record<StudioId, StudioDeepSupplement> = {
  ...DEEP_SUPPLEMENTS_PART_1,
  ...DEEP_SUPPLEMENTS_PART_2,
};

function enrichChapter(ch: StudioChapter): StudioChapter {
  const sup = ALL_SUPPLEMENTS[ch.studioId];
  if (!sup) return ch;
  return {
    ...ch,
    methods: sup.extraMethods?.length ? [...ch.methods, ...sup.extraMethods] : ch.methods,
    deepTheory: sup.deepTheory,
    workedExamples: sup.workedExamples,
    uiControlsGuide: sup.uiControlsGuide,
    troubleshootingPlaybook: sup.troubleshootingPlaybook,
  };
}

export const ALL_STUDIO_CHAPTERS: StudioChapter[] = [
  ...HANDBOOK_CHAPTERS_PART1,
  ...HANDBOOK_CHAPTERS_PART2,
].map(enrichChapter);

export const CHAPTER_BY_STUDIO: Record<StudioId, StudioChapter> = ALL_STUDIO_CHAPTERS.reduce(
  (acc, ch) => {
    acc[ch.studioId] = ch;
    return acc;
  },
  {} as Record<StudioId, StudioChapter>,
);

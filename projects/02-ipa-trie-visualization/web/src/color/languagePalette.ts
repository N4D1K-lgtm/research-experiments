import type { PhonologicalPosition } from "../types";

/** Phonological role color config */
export interface RoleInfo {
  position: PhonologicalPosition;
  label: string;
  hue: number;
  color: string;
}

export const ROLES: RoleInfo[] = [
  { position: "onset",   label: "Onset",   hue: 200, color: "#39a5c9" },
  { position: "nucleus", label: "Nucleus", hue: 45,  color: "#c9a539" },
  { position: "coda",    label: "Coda",    hue: 280, color: "#9539c9" },
  { position: "mixed",   label: "Mixed",   hue: 150, color: "#39c980" },
];

export function getRoleColor(position: PhonologicalPosition): string {
  return ROLES.find((r) => r.position === position)?.color ?? "#888888";
}

export function getRoleHue(position: PhonologicalPosition): number {
  return ROLES.find((r) => r.position === position)?.hue ?? 150;
}

export function getRoleLabel(position: PhonologicalPosition): string {
  return ROLES.find((r) => r.position === position)?.label ?? position;
}

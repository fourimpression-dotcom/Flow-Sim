export function requireElement<T extends Element>(selector: string): T {
  const el = document.querySelector<T>(selector);
  if (!el) {
    throw new Error(`Element not found: ${selector}`);
  }
  return el;
}

export function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

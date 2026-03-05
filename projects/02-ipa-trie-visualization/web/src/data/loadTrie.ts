import type { TrieData } from "../types";

export async function loadTrie(url = "/trie.json"): Promise<TrieData> {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to load trie: ${resp.status}`);
  return resp.json();
}

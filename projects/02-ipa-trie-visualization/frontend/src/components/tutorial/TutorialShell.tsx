import { useState, useEffect, useRef } from "react";
import { COLORS, FONTS } from "../../styles/theme";
import { ChapterHero } from "./chapters/ChapterHero";
import { ChapterSound } from "./chapters/ChapterSound";
import { ChapterTokenization } from "./chapters/ChapterTokenization";
import { ChapterTrie } from "./chapters/ChapterTrie";
import { ChapterScale } from "./chapters/ChapterScale";
import { ChapterEntropy } from "./chapters/ChapterEntropy";
import { ChapterTransitions } from "./chapters/ChapterTransitions";
import { ChapterCrossLinguistic } from "./chapters/ChapterCrossLinguistic";
import { ChapterExplorer } from "./chapters/ChapterExplorer";

const CHAPTERS = [
  { id: "hero", label: "·" },
  { id: "sound", label: "Sound" },
  { id: "tokenization", label: "Tokens" },
  { id: "trie", label: "Trie" },
  { id: "scale", label: "Scale" },
  { id: "entropy", label: "Entropy" },
  { id: "transitions", label: "Transitions" },
  { id: "crossling", label: "Languages" },
  { id: "explorer", label: "Explorer" },
];

export function TutorialShell() {
  const [activeChapter, setActiveChapter] = useState("hero");
  const [scrollProgress, setScrollProgress] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleScroll = () => {
      const scrollTop = container.scrollTop;
      const scrollHeight = container.scrollHeight - container.clientHeight;
      setScrollProgress(scrollHeight > 0 ? scrollTop / scrollHeight : 0);
    };

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting && entry.intersectionRatio > 0.3) {
            setActiveChapter(entry.target.id);
          }
        }
      },
      { root: container, threshold: 0.3 },
    );

    const sections = container.querySelectorAll("section[id]");
    sections.forEach((s) => observer.observe(s));
    container.addEventListener("scroll", handleScroll, { passive: true });

    return () => {
      sections.forEach((s) => observer.unobserve(s));
      container.removeEventListener("scroll", handleScroll);
    };
  }, []);

  const scrollTo = (id: string) => {
    const el = document.getElementById(id);
    el?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div style={{ position: "fixed", inset: 0, background: COLORS.bg, display: "flex" }}>
      {/* Progress bar */}
      <div
        style={{
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          height: 2,
          zIndex: 100,
          background: "rgba(255,255,255,0.03)",
        }}
      >
        <div
          style={{
            height: "100%",
            width: `${scrollProgress * 100}%`,
            background: `linear-gradient(90deg, ${COLORS.accent}88, ${COLORS.accent})`,
            transition: "width 0.1s ease",
          }}
        />
      </div>

      {/* Nav dots */}
      <nav
        style={{
          position: "fixed",
          right: 16,
          top: "50%",
          transform: "translateY(-50%)",
          zIndex: 90,
          display: "flex",
          flexDirection: "column",
          alignItems: "flex-end",
          gap: 8,
        }}
      >
        {CHAPTERS.map((ch) => (
          <button
            key={ch.id}
            onClick={() => scrollTo(ch.id)}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              background: "none",
              border: "none",
              cursor: "pointer",
              padding: "2px 0",
              flexDirection: "row-reverse",
            }}
          >
            <div
              style={{
                width: activeChapter === ch.id ? 8 : 5,
                height: activeChapter === ch.id ? 8 : 5,
                borderRadius: "50%",
                background: activeChapter === ch.id ? COLORS.accent : "rgba(255,255,255,0.15)",
                transition: "all 0.2s ease",
                border: activeChapter === ch.id ? `1px solid ${COLORS.accent}` : "1px solid transparent",
              }}
            />
            <span
              style={{
                fontSize: 8,
                fontFamily: FONTS.mono,
                color: activeChapter === ch.id ? COLORS.text : "transparent",
                textTransform: "uppercase",
                letterSpacing: 1,
                transition: "color 0.2s ease",
                whiteSpace: "nowrap",
              }}
            >
              {ch.label}
            </span>
          </button>
        ))}
      </nav>

      {/* Scroll container */}
      <div
        ref={containerRef}
        style={{
          flex: 1,
          overflowY: "auto",
          overflowX: "hidden",
        }}
      >
        <ChapterHero />
        <ChapterSound />
        <ChapterTokenization />
        <ChapterTrie />
        <ChapterScale />
        <ChapterEntropy />
        <ChapterTransitions />
        <ChapterCrossLinguistic />
        <ChapterExplorer />
      </div>
    </div>
  );
}

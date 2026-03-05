import { useState, useRef, useEffect } from "react";
import { COLORS, FONTS } from "../styles/theme";

interface Props {
  text: string;
  maxWidth?: number;
}

export function InfoTooltip({ text, maxWidth = 260 }: Props) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<"above" | "below">("below");
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (open && ref.current) {
      const rect = ref.current.getBoundingClientRect();
      if (rect.bottom > window.innerHeight - 20) setPos("above");
      else setPos("below");
    }
  }, [open]);

  return (
    <span
      style={{ position: "relative", display: "inline-flex", alignItems: "center", marginLeft: 6 }}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
    >
      <span
        style={{
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
          width: 14,
          height: 14,
          borderRadius: "50%",
          background: "rgba(255,255,255,0.06)",
          border: `1px solid ${COLORS.border}`,
          fontSize: 9,
          fontWeight: 600,
          color: COLORS.textDim,
          cursor: "help",
          fontFamily: FONTS.sans,
          lineHeight: 1,
          flexShrink: 0,
        }}
      >
        i
      </span>
      {open && (
        <div
          ref={ref}
          style={{
            position: "absolute",
            left: "50%",
            transform: "translateX(-50%)",
            ...(pos === "below"
              ? { top: "calc(100% + 8px)" }
              : { bottom: "calc(100% + 8px)" }),
            width: maxWidth,
            padding: "10px 14px",
            background: "rgba(10, 10, 18, 0.96)",
            border: `1px solid ${COLORS.borderLight}`,
            borderRadius: 8,
            backdropFilter: "blur(16px)",
            boxShadow: "0 8px 32px rgba(0,0,0,0.6)",
            zIndex: 200,
            fontSize: 11,
            lineHeight: 1.6,
            color: COLORS.text,
            fontFamily: FONTS.sans,
            fontWeight: 400,
            pointerEvents: "none",
          }}
        >
          {text}
        </div>
      )}
    </span>
  );
}

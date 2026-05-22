"use client";

import dynamic from "next/dynamic";

export const DrawingFlow = dynamic(
  () => import("./drawing-flow-impl").then((m) => m.DrawingFlow),
  {
    ssr: false,
    loading: () => (
      <div className="h-full grid place-items-center bg-stone-100 text-sm text-stone-500">
        Lade Karte…
      </div>
    ),
  },
);

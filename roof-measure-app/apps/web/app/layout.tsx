import type { Metadata } from "next";
import "leaflet/dist/leaflet.css";
import "./globals.css";

export const metadata: Metadata = {
  title: "Dach vermessen",
  description:
    "Zeichne dein Dach auf dem Luftbild ein und erhalte sofort Fläche, Neigung und eine Kostenschätzung für die Sanierung.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="de">
      <body>{children}</body>
    </html>
  );
}

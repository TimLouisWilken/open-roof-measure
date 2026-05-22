import { redirect } from "next/navigation";

import { DrawingFlow } from "@/components/drawing-flow";
import { geocode } from "@/lib/api";

type Props = {
  searchParams: Promise<{ q?: string }>;
};

export default async function MeasurePage({ searchParams }: Props) {
  const params = await searchParams;
  const query = params.q?.trim();
  if (!query) {
    redirect("/");
  }

  const result = await geocode(query);

  if (!result) {
    return (
      <main className="min-h-screen grid place-items-center px-6">
        <div className="text-center space-y-4 max-w-md">
          <h1 className="text-2xl font-semibold">Adresse nicht gefunden</h1>
          <p className="text-stone-600">
            Wir konnten <strong>{query}</strong> nicht eindeutig zuordnen. Bitte gib
            Straße, Hausnummer und Stadt vollständig ein.
          </p>
          <a href="/" className="inline-block text-amber-700 hover:underline">
            Zurück zur Startseite
          </a>
        </div>
      </main>
    );
  }

  return (
    <main className="h-screen flex flex-col">
      <header className="border-b border-stone-200 bg-white px-4 py-3 flex items-center gap-4">
        <a href="/" className="text-sm text-stone-500 hover:text-stone-800">←</a>
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium truncate">{result.display_name}</p>
          <p className="text-xs text-stone-500">
            {result.lat.toFixed(5)}, {result.lon.toFixed(5)} · {result.state ?? "—"}
          </p>
        </div>
      </header>

      <div className="flex-1 min-h-0">
        <DrawingFlow
          lat={result.lat}
          lon={result.lon}
          address={result.display_name}
        />
      </div>
    </main>
  );
}

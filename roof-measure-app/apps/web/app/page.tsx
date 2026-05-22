import Link from "next/link";

export default function Home() {
  return (
    <main className="min-h-screen flex flex-col">
      <header className="border-b border-stone-200 bg-white">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
          <span className="font-semibold tracking-tight">Dach vermessen</span>
          <nav className="flex gap-6 text-sm text-stone-600">
            <Link href="/wie-es-funktioniert">So funktioniert es</Link>
            <Link href="/datenschutz">Datenschutz</Link>
          </nav>
        </div>
      </header>

      <section className="flex-1 grid place-items-center px-6 py-20">
        <div className="max-w-2xl text-center space-y-6">
          <h1 className="text-4xl sm:text-5xl font-semibold tracking-tight leading-tight">
            Wie groß ist dein Dach wirklich?
          </h1>
          <p className="text-lg text-stone-600">
            Adresse eingeben, dein Dach im Luftbild einzeichnen, sofort Fläche, Neigung und eine
            Kostenschätzung erhalten — kostenlos und ohne Vor-Ort-Termin.
          </p>

          <form action="/messen" className="flex gap-2 max-w-xl mx-auto pt-4">
            <input
              name="q"
              required
              minLength={3}
              type="text"
              placeholder="Straße, Hausnummer, Stadt"
              className="flex-1 rounded-md border border-stone-300 px-4 py-3 text-base focus:outline-none focus:border-amber-600 bg-white"
            />
            <button
              type="submit"
              className="rounded-md bg-amber-600 hover:bg-amber-700 text-white px-5 py-3 font-medium"
            >
              Loslegen
            </button>
          </form>

          <p className="text-xs text-stone-500 pt-4">
            Aktuell verfügbar in Nordrhein-Westfalen. Weitere Bundesländer in Kürze.
          </p>
        </div>
      </section>

      <footer className="border-t border-stone-200 bg-white">
        <div className="max-w-5xl mx-auto px-6 py-4 text-xs text-stone-500 flex gap-6">
          <span>© {new Date().getFullYear()} roof-measure</span>
          <Link href="/impressum">Impressum</Link>
          <Link href="/datenschutz">Datenschutz</Link>
        </div>
      </footer>
    </main>
  );
}

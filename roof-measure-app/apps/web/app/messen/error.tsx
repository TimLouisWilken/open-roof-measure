"use client";

export default function MessenError({ error, reset }: { error: Error & { digest?: string }; reset: () => void }) {
  return (
    <main className="min-h-screen grid place-items-center px-6 py-12">
      <div className="max-w-2xl space-y-4">
        <h1 className="text-2xl font-semibold text-red-700">Fehler in /messen</h1>
        <pre className="text-sm bg-red-50 border border-red-200 rounded p-4 overflow-auto whitespace-pre-wrap break-all">
          <strong>Message:</strong> {error.message}
          {error.stack ? `\n\n${error.stack}` : ""}
          {error.digest ? `\n\nDigest: ${error.digest}` : ""}
        </pre>
        <button
          onClick={reset}
          className="rounded-md bg-stone-800 text-white px-4 py-2 hover:bg-stone-700"
        >
          Erneut versuchen
        </button>
      </div>
    </main>
  );
}

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type GeocodeResult = {
  query: string;
  display_name: string;
  lat: number;
  lon: number;
  state: string | null;
  postcode: string | null;
  country_code: string | null;
};

export async function geocode(query: string): Promise<GeocodeResult | null> {
  const url = new URL("/geocode", API_URL);
  url.searchParams.set("q", query);

  const response = await fetch(url, { cache: "no-store" });
  if (response.status === 404) return null;
  if (!response.ok) {
    throw new Error(`Geocoding failed: ${response.status}`);
  }
  return (await response.json()) as GeocodeResult;
}

export type ElevationResponse = {
  heights: (number | null)[];
};

export async function sampleElevations(
  points: [number, number][], // [lat, lon]
): Promise<ElevationResponse> {
  const response = await fetch(`${API_URL}/elevation/sample`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ points }),
  });
  if (!response.ok) {
    throw new Error(`Elevation sampling failed: ${response.status}`);
  }
  return (await response.json()) as ElevationResponse;
}

export type RoofSection = {
  points: [number, number][];
  area_m2_2d: number;
  pitch_deg: number | null;
  aspect_deg: number | null;
  aspect_label: string | null;
  area_m2_slope: number | null;
  mean_height_m: number | null;
  min_height_m: number | null;
  max_height_m: number | null;
  n_samples: number;
};

export type SectionsResponse = {
  sections: RoofSection[];
};

export async function computeSections(
  polygon: [number, number][],
  ridges: [number, number][][],
  valleys: [number, number][][],
): Promise<SectionsResponse> {
  const response = await fetch(`${API_URL}/roof/sections`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ polygon, ridges, valleys }),
  });
  if (!response.ok) {
    throw new Error(`Section split failed: ${response.status}`);
  }
  return (await response.json()) as SectionsResponse;
}

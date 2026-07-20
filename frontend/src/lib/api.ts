const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8080/api";

export type UserOut = {
  id: number;
  email: string;
  name: string;
  created_at: string;
};

export type TokenPair = {
  access_token: string;
  refresh_token: string;
  token_type: string;
  user: UserOut;
};

export type FaceResult = {
  label: string;
  real_confidence: number;
  fake_confidence: number;
  threat: string;
  image_url?: string | null;
};

export type TimingMs = {
  detect: number;
  classify: number;
  total: number;
};

export type ImageDetectionResult = {
  id: number | null;
  prediction: string;
  confidence: number;
  face_count: number;
  faces: FaceResult[];
  processed_image_url?: string | null;
  original_image_url: string;
  timing_ms?: TimingMs | null;
  created_at: string;
};

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

export function friendlyErrorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  if (err instanceof TypeError) {
    return `Could not reach the server at ${API_BASE}. Is the backend running?`;
  }
  return "Something went wrong";
}

async function parseError(res: Response): Promise<string> {
  try {
    const data = await res.json();
    return data.detail || data.message || res.statusText;
  } catch {
    return res.statusText;
  }
}

async function request<T>(
  path: string,
  init: RequestInit & { auth?: boolean } = {}
): Promise<T> {
  const { auth, headers, ...rest } = init;
  const finalHeaders = new Headers(headers);

  if (auth) {
    const token = getAccessToken();
    if (token) finalHeaders.set("Authorization", `Bearer ${token}`);
  }

  const res = await fetch(`${API_BASE}${path}`, { ...rest, headers: finalHeaders });

  if (!res.ok) {
    throw new ApiError(res.status, await parseError(res));
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

// ── token storage ────────────────────────────────────────────────────────

const ACCESS_KEY = "deeptrace_access_token";
const REFRESH_KEY = "deeptrace_refresh_token";
const USER_KEY = "deeptrace_user";

export function getAccessToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(ACCESS_KEY);
}

export function getRefreshToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.localStorage.getItem(REFRESH_KEY);
}

export function getStoredUser(): UserOut | null {
  if (typeof window === "undefined") return null;
  const raw = window.localStorage.getItem(USER_KEY);
  return raw ? (JSON.parse(raw) as UserOut) : null;
}

export function storeSession(tokens: TokenPair) {
  window.localStorage.setItem(ACCESS_KEY, tokens.access_token);
  window.localStorage.setItem(REFRESH_KEY, tokens.refresh_token);
  window.localStorage.setItem(USER_KEY, JSON.stringify(tokens.user));
}

export function clearSession() {
  window.localStorage.removeItem(ACCESS_KEY);
  window.localStorage.removeItem(REFRESH_KEY);
  window.localStorage.removeItem(USER_KEY);
}

// ── auth ─────────────────────────────────────────────────────────────────

export function register(email: string, password: string, name: string) {
  return request<TokenPair>("/auth/register", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, name }),
  });
}

export function login(email: string, password: string) {
  return request<TokenPair>("/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
}

export function refreshAccessToken(refresh_token: string) {
  return request<{ access_token: string; token_type: string }>("/auth/refresh", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ refresh_token }),
  });
}

export function me() {
  return request<UserOut>("/auth/me", { auth: true });
}

// ── detection ────────────────────────────────────────────────────────────

export function detectImage(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  return request<ImageDetectionResult>("/detection/image", {
    method: "POST",
    body: formData,
    auth: true,
  });
}

// ── generation ───────────────────────────────────────────────────────────

export type SwapResult = {
  success: boolean;
  job_id: string;
  output_filename: string;
  output_url: string;
  output_type: string;
  processing_time: number;
  faces_swapped: number;
  preview_b64?: string;
  error?: string | null;
};

export function genSwap(
  source: File,
  target: File,
  options: Partial<{
    face_indices: string;
    enhance: string;
    pitch_semitones: number;
    detect_interval: number;
    max_side: number;
    async_mode: boolean;
  }> = {}
) {
  const formData = new FormData();
  formData.append("source", source);
  formData.append("target", target);
  formData.append("face_indices", options.face_indices ?? "");
  formData.append("enhance", options.enhance ?? "1");
  formData.append("pitch_semitones", String(options.pitch_semitones ?? 0));
  formData.append("detect_interval", String(options.detect_interval ?? 5));
  formData.append("max_side", String(options.max_side ?? 720));
  formData.append("async_mode", options.async_mode ? "true" : "false");

  return request<SwapResult>("/generation/swap", {
    method: "POST",
    body: formData,
    auth: true,
  });
}

export type SwapProgress = {
  found: boolean;
  total?: number;
  done?: number;
  status?: string;
  eta_seconds?: number;
  fps_proc?: number;
  skipped?: number;
};

export function genProgress(jobId: string) {
  return request<SwapProgress>(`/generation/progress/${jobId}`, { auth: true });
}

export function genResult(jobId: string) {
  return request<SwapResult>(`/generation/result/${jobId}`, { auth: true });
}

/** Poll a video swap job until it finishes, reporting progress along the way.
 * Video swaps run async on the engine (frame-by-frame, CPU-bound — can take
 * well over a minute) so the caller gets a job_id immediately and this polls
 * real per-frame progress instead of blocking on one long request with no
 * feedback. */
export async function pollSwapJob(
  jobId: string,
  onProgress: (p: SwapProgress) => void,
  { intervalMs = 700, timeoutMs = 20 * 60 * 1000 }: { intervalMs?: number; timeoutMs?: number } = {}
): Promise<SwapResult> {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    const p = await genProgress(jobId);
    onProgress(p);
    if (p.status === "done" || p.status === "error") {
      // The engine flips progress to "done" a moment before it finishes
      // writing the final result — a single-shot fetch right after can lose
      // that race and 404. Retry briefly to close the window.
      for (let attempt = 0; attempt < 6; attempt++) {
        try {
          return await genResult(jobId);
        } catch (err) {
          if (attempt === 5 || !(err instanceof ApiError) || err.status !== 404) throw err;
          await new Promise((resolve) => setTimeout(resolve, 400));
        }
      }
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  throw new ApiError(408, "Generation timed out — the job may still be running on the server.");
}

/** Resolve a swap result to a displayable, type-aware output. Images get an
 * inline base64 preview when available (correct regardless of static-file
 * routing); video never has a preview (too large), so it always goes through
 * the backend's output-proxy route and must be rendered with <video>, not
 * <img> — mixing the two is exactly what produced a broken-image icon for
 * video swaps. */
export function swapOutputSrc(
  res: SwapResult
): { type: "image" | "video"; src: string } | null {
  if (res.output_type === "video") {
    if (!res.output_filename) return null;
    return { type: "video", src: `${apiBase()}/generation/output/${res.output_filename}` };
  }
  if (res.preview_b64) {
    return { type: "image", src: `data:image/jpeg;base64,${res.preview_b64}` };
  }
  if (res.output_filename) {
    return { type: "image", src: `${apiBase()}/generation/output/${res.output_filename}` };
  }
  return null;
}

export type FaceCrop = {
  b64: string;
  score: number;
  bbox: number[];
};

export type DetectFacesResult = {
  success: boolean;
  count: number;
  faces: FaceCrop[];
  image_path?: string;
};

export function genDetectFaces(image: File) {
  const formData = new FormData();
  formData.append("image", image);
  return request<DetectFacesResult>("/generation/detect-faces", {
    method: "POST",
    body: formData,
    auth: true,
  });
}

export function apiBase() {
  return API_BASE;
}

export function mediaUrl(path?: string | null) {
  if (!path) return null;
  if (path.startsWith("http")) return path;
  const origin = API_BASE.replace(/\/api\/?$/, "");
  return `${origin}${path}`;
}

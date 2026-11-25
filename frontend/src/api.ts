const SESSION_KEY = 'omniforge_session_id';
const CSRF_COOKIE_NAME = 'omniforge_csrf';
const API_KEY_STORAGE = 'omniforge_api_key';
const BEARER_STORAGE = 'omniforge_bearer_token';

/** Default per-verb timeouts (ms). Long-running modeling jobs pass an explicit override. */
export const DEFAULT_TIMEOUTS = {
  get: 30_000,
  post: 300_000,
  delete: 30_000,
  upload: 120_000,
  download: 120_000,
} as const;

/** Typed API error so callers can distinguish timeouts, aborts, and HTTP failures. */
export class ApiError extends Error {
  readonly status: number;
  readonly retryable: boolean;
  readonly code: 'TIMEOUT' | 'ABORTED' | 'NETWORK' | 'HTTP' | 'PARSE';

  constructor(message: string, status: number, code: ApiError['code']) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    // Retry only transient failures: timeouts, network drops, and 5xx / 429.
    this.retryable = code === 'TIMEOUT' || code === 'NETWORK' || status === 429 || (status >= 500 && status < 600);
  }
}

export interface ApiRequestOptions {
  /** Per-request deadline in milliseconds. Defaults depend on the verb. */
  timeoutMs?: number;
  /** Caller-owned cancellation signal (e.g. from a component unmount). */
  signal?: AbortSignal;
  /** Max retries for transient failures. GET defaults to 2; other verbs default to 0. */
  retries?: number;
}

/** Loose JSON object payload returned by most workbench endpoints. */
export type JsonObject = Record<string, unknown>;

/** Extract a human-readable message from an unknown thrown value. */
export function toErrorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  if (typeof err === 'string') return err;
  if (err !== null && typeof err === 'object' && 'message' in err) {
    const msg = (err as { message?: unknown }).message;
    if (typeof msg === 'string' && msg) return msg;
  }
  return String(err);
}

/** Narrow an unknown value to a JSON object (empty object when not an object). */
export function asJsonObject(value: unknown): JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? (value as JsonObject)
    : {};
}

/** Narrow an unknown value to an array (empty array when not an array). */
export function asJsonArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

/** Read a string field from a JSON object with a fallback. */
export function getString(obj: JsonObject, key: string, fallback = ''): string {
  const v = obj[key];
  return typeof v === 'string' ? v : v === null || v === undefined ? fallback : String(v);
}

/** Read a numeric field from a JSON object with a fallback. */
export function getNumber(obj: JsonObject, key: string, fallback = 0): number {
  const v = obj[key];
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}

function generateSecureSessionId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return 'of_' + crypto.randomUUID().replace(/-/g, '').slice(0, 16);
  }
  if (typeof crypto !== 'undefined' && typeof crypto.getRandomValues === 'function') {
    const bytes = new Uint8Array(12);
    crypto.getRandomValues(bytes);
    return 'of_' + Array.from(bytes, (b) => b.toString(16).padStart(2, '0')).join('');
  }
  return 'of_' + Date.now().toString(36);
}

export function getSessionId(): string {
  let sid = localStorage.getItem(SESSION_KEY);
  if (!sid) {
    sid = generateSecureSessionId();
    localStorage.setItem(SESSION_KEY, sid);
  }
  return sid;
}

function getCookieValue(name: string): string {
  if (typeof document === 'undefined' || !document.cookie) {
    return '';
  }
  const prefix = `${name}=`;
  for (const part of document.cookie.split(';')) {
    const trimmed = part.trim();
    if (trimmed.startsWith(prefix)) {
      return decodeURIComponent(trimmed.slice(prefix.length));
    }
  }
  return '';
}

function buildAuthHeaders(extra?: Record<string, string>): Record<string, string> {
  const headers: Record<string, string> = {
    'X-Session-ID': getSessionId(),
    ...(extra || {}),
  };
  const csrfToken = getCookieValue(CSRF_COOKIE_NAME);
  if (csrfToken) {
    headers['X-CSRF-Token'] = csrfToken;
  }
  const apiKey = localStorage.getItem(API_KEY_STORAGE);
  if (apiKey) {
    headers['X-API-Key'] = apiKey;
  }
  const bearer = localStorage.getItem(BEARER_STORAGE);
  if (bearer) {
    headers['Authorization'] = `Bearer ${bearer}`;
  }
  return headers;
}

export function hasStoredCredentials(): boolean {
  if (typeof localStorage === 'undefined') return false;
  return Boolean(localStorage.getItem(API_KEY_STORAGE) || localStorage.getItem(BEARER_STORAGE));
}

export function setStoredCredentials(creds: { apiKey?: string; bearerToken?: string }): void {
  const apiKey = (creds.apiKey ?? '').trim();
  const bearer = (creds.bearerToken ?? '').trim();
  if (apiKey) localStorage.setItem(API_KEY_STORAGE, apiKey);
  else localStorage.removeItem(API_KEY_STORAGE);
  if (bearer) localStorage.setItem(BEARER_STORAGE, bearer);
  else localStorage.removeItem(BEARER_STORAGE);
}

export function clearStoredCredentials(): void {
  localStorage.removeItem(API_KEY_STORAGE);
  localStorage.removeItem(BEARER_STORAGE);
}

let inFlightCookieRefresh: Promise<boolean> | null = null;

async function refreshWorkbenchSessionCookie(): Promise<boolean> {
  if (typeof window === 'undefined' || hasStoredCredentials()) {
    return false;
  }
  if (inFlightCookieRefresh) {
    return inFlightCookieRefresh;
  }
  inFlightCookieRefresh = (async () => {
    const prevCsrf = getCookieValue(CSRF_COOKIE_NAME);
    try {
      const res = await fetch('/', {
        method: 'GET',
        credentials: 'same-origin',
        cache: 'no-store',
      });
      if (!res.ok) return false;
      const nextCsrf = getCookieValue(CSRF_COOKIE_NAME);
      return Boolean(nextCsrf && nextCsrf !== prevCsrf);
    } catch {
      return false;
    } finally {
      inFlightCookieRefresh = null;
    }
  })();
  return inFlightCookieRefresh;
}

interface ValidationDetailItem {
  loc?: unknown;
  msg?: unknown;
}

function formatValidationDetails(items: ValidationDetailItem[]): string {
  return items
    .map((item) => {
      const loc = Array.isArray(item?.loc) ? item.loc.map(String).join('.') : '';
      const msg = typeof item?.msg === 'string' ? item.msg : JSON.stringify(item);
      return loc ? `${loc}: ${msg}` : msg;
    })
    .join('; ');
}

function extractErrorMessage(rawText: string, status: number, statusText: string): string {
  const trimmed = rawText.trim();
  if (trimmed) {
    try {
      const data: unknown = JSON.parse(trimmed);
      if (typeof data === 'object' && data !== null) {
        const record = data as Record<string, unknown>;
        if (typeof record.detail === 'string' && record.detail.trim()) {
          return record.detail;
        }
        if (Array.isArray(record.detail)) {
          const joined = formatValidationDetails(record.detail as ValidationDetailItem[]);
          if (joined) return joined;
        }
        if (typeof record.message === 'string' && record.message.trim()) {
          return typeof record.error_type === 'string' && record.error_type
            ? `${record.error_type}: ${record.message}`
            : record.message;
        }
        if (typeof record.error === 'string' && record.error.trim()) {
          return record.error;
        }
      }
      return JSON.stringify(data);
    } catch {
      return trimmed.length > 400 ? `${trimmed.slice(0, 400)}...` : trimmed;
    }
  }
  return `HTTP ${status}${statusText ? ` ${statusText}` : ''}`;
}

/**
 * Combine a caller signal with a deadline. Prefers AbortSignal.timeout/any
 * (modern browsers) and falls back to a manual AbortController timer.
 */
function combineSignals(callerSignal: AbortSignal | undefined, timeoutMs: number): AbortSignal {
  const timeoutSignal: AbortSignal =
    typeof AbortSignal.timeout === 'function'
      ? AbortSignal.timeout(timeoutMs)
      : (() => {
          const controller = new AbortController();
          const timer = window.setTimeout(() => controller.abort(), timeoutMs);
          controller.signal.addEventListener('abort', () => window.clearTimeout(timer), { once: true });
          return controller.signal;
        })();
  if (!callerSignal) return timeoutSignal;
  if (typeof AbortSignal.any === 'function') {
    return AbortSignal.any([callerSignal, timeoutSignal]);
  }
  const controller = new AbortController();
  const forward = (): void => controller.abort(callerSignal.reason ?? timeoutSignal.reason);
  if (callerSignal.aborted || timeoutSignal.aborted) {
    forward();
    return controller.signal;
  }
  callerSignal.addEventListener('abort', forward, { once: true });
  timeoutSignal.addEventListener('abort', forward, { once: true });
  return controller.signal;
}

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && (err.name === 'AbortError' || err.name === 'TimeoutError');
}

function toApiError(err: unknown, path: string, timeoutMs: number): ApiError {
  if (err instanceof ApiError) return err;
  if (isAbortError(err)) {
    const name = err instanceof DOMException ? err.name : 'AbortError';
    if (name === 'TimeoutError') {
      return new ApiError(`Request to ${path} timed out after ${timeoutMs} ms.`, 0, 'TIMEOUT');
    }
    return new ApiError(`Request to ${path} was aborted.`, 0, 'ABORTED');
  }
  if (err instanceof TypeError) {
    return new ApiError(`Network error while contacting ${path}: ${err.message}`, 0, 'NETWORK');
  }
  return new ApiError(err instanceof Error ? err.message : String(err), 0, 'NETWORK');
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException('Aborted', 'AbortError'));
      return;
    }
    const timer = window.setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    const onAbort = (): void => {
      window.clearTimeout(timer);
      reject(new DOMException('Aborted', 'AbortError'));
    };
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

async function fetchWithRetry(
  path: string,
  init: Omit<RequestInit, 'headers'>,
  extraHeaders: Record<string, string> | undefined,
  options: ApiRequestOptions,
  defaultTimeoutMs: number,
  defaultRetries: number,
): Promise<Response> {
  const timeoutMs = options.timeoutMs ?? defaultTimeoutMs;
  const maxRetries = options.retries ?? defaultRetries;
  let attempt = 0;
  let refreshedSession = false;
  for (;;) {
    const signal = combineSignals(options.signal, timeoutMs);
    try {
      const res = await fetch(path, {
        ...init,
        headers: buildAuthHeaders(extraHeaders),
        signal,
      });
      if (res.status === 401 && !refreshedSession && path.startsWith('/api/')) {
        refreshedSession = true;
        if (await refreshWorkbenchSessionCookie()) {
          continue;
        }
      }
      // Retry transient HTTP statuses only when retries remain.
      if ((res.status === 429 || (res.status >= 500 && res.status < 600)) && attempt < maxRetries) {
        const rawText = await res.text();
        const message = extractErrorMessage(rawText, res.status, res.statusText);
        const err = new ApiError(message, res.status, 'HTTP');
        if (!err.retryable) throw err;
        attempt += 1;
        await sleep(Math.min(1000 * 2 ** (attempt - 1), 4000) + Math.random() * 250, options.signal);
        continue;
      }
      return res;
    } catch (err) {
      const apiErr = toApiError(err, path, timeoutMs);
      if (apiErr.code === 'ABORTED' || !apiErr.retryable || attempt >= maxRetries) {
        throw apiErr;
      }
      attempt += 1;
      await sleep(Math.min(1000 * 2 ** (attempt - 1), 4000) + Math.random() * 250, options.signal);
    }
  }
}

async function handleResponse<T>(res: Response): Promise<T> {
  // Always read the Response body stream ONCE to avoid "body stream already read" errors.
  const rawText = await res.text();
  if (!res.ok) {
    throw new ApiError(extractErrorMessage(rawText, res.status, res.statusText), res.status, 'HTTP');
  }
  const trimmed = rawText.trim();
  if (!trimmed) {
    return {} as T;
  }
  try {
    return JSON.parse(trimmed) as T;
  } catch {
    throw new ApiError(
      `Unexpected non-JSON response from server (HTTP ${res.status}): ${trimmed.slice(0, 200)}`,
      res.status,
      'PARSE',
    );
  }
}

export async function apiGet<T = unknown>(path: string, options: ApiRequestOptions = {}): Promise<T> {
  const res = await fetchWithRetry(
    path,
    { credentials: 'same-origin' },
    undefined,
    options,
    DEFAULT_TIMEOUTS.get,
    2,
  );
  return handleResponse<T>(res);
}

export async function apiPost<T = unknown>(
  path: string,
  body?: unknown,
  options: ApiRequestOptions = {},
): Promise<T> {
  const res = await fetchWithRetry(
    path,
    {
      method: 'POST',
      credentials: 'same-origin',
      body: body === undefined ? '{}' : JSON.stringify(body),
    },
    { 'Content-Type': 'application/json' },
    options,
    DEFAULT_TIMEOUTS.post,
    0,
  );
  return handleResponse<T>(res);
}

export async function apiDelete<T = unknown>(path: string, options: ApiRequestOptions = {}): Promise<T> {
  const res = await fetchWithRetry(
    path,
    { method: 'DELETE', credentials: 'same-origin' },
    undefined,
    options,
    DEFAULT_TIMEOUTS.delete,
    0,
  );
  return handleResponse<T>(res);
}

export async function apiUpload<T = unknown>(
  path: string,
  formData: FormData,
  options: ApiRequestOptions = {},
): Promise<T> {
  const res = await fetchWithRetry(
    path,
    { method: 'POST', credentials: 'same-origin', body: formData },
    undefined,
    options,
    DEFAULT_TIMEOUTS.upload,
    0,
  );
  return handleResponse<T>(res);
}

export async function apiDownload(
  path: string,
  filename: string,
  method: 'GET' | 'POST' = 'GET',
  formData?: FormData,
  options: ApiRequestOptions = {},
): Promise<void> {
  const res = await fetchWithRetry(
    path,
    { method, credentials: 'same-origin', body: formData },
    undefined,
    options,
    DEFAULT_TIMEOUTS.download,
    method === 'GET' ? 1 : 0,
  );
  if (!res.ok) {
    const rawText = await res.text();
    throw new ApiError(extractErrorMessage(rawText, res.status, res.statusText), res.status, 'HTTP');
  }
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

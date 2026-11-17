const SESSION_KEY = 'omniforge_session_id';

export function getSessionId(): string {
  let sid = localStorage.getItem(SESSION_KEY);
  if (!sid) {
    sid = 'of_' + Math.random().toString(36).slice(2, 12);
    localStorage.setItem(SESSION_KEY, sid);
  }
  return sid;
}

function extractErrorMessage(rawText: string, status: number, statusText: string): string {
  const trimmed = rawText.trim();
  if (trimmed) {
    try {
      const data = JSON.parse(trimmed);
      if (typeof data?.detail === 'string' && data.detail.trim()) {
        return data.detail;
      }
      if (Array.isArray(data?.detail)) {
        const joined = data.detail
          .map((item: any) => {
            const loc = Array.isArray(item?.loc) ? item.loc.join('.') : '';
            return loc ? `${loc}: ${item?.msg || JSON.stringify(item)}` : (item?.msg || JSON.stringify(item));
          })
          .join('; ');
        if (joined) return joined;
      }
      if (typeof data?.message === 'string' && data.message.trim()) {
        return data.error_type ? `${data.error_type}: ${data.message}` : data.message;
      }
      if (typeof data?.error === 'string' && data.error.trim()) {
        return data.error;
      }
      return JSON.stringify(data);
    } catch {
      return trimmed.length > 400 ? `${trimmed.slice(0, 400)}...` : trimmed;
    }
  }
  return `HTTP ${status}${statusText ? ` ${statusText}` : ''}`;
}

async function handleResponse<T>(res: Response): Promise<T> {
  // Always read the Response body stream ONCE to avoid "body stream already read" errors.
  const rawText = await res.text();
  if (!res.ok) {
    throw new Error(extractErrorMessage(rawText, res.status, res.statusText));
  }
  const trimmed = rawText.trim();
  if (!trimmed) {
    return {} as T;
  }
  try {
    return JSON.parse(trimmed) as T;
  } catch {
    throw new Error(`Unexpected non-JSON response from server (HTTP ${res.status}): ${trimmed.slice(0, 200)}`);
  }
}

export async function apiGet<T = any>(path: string): Promise<T> {
  const res = await fetch(path, {
    headers: { 'X-Session-ID': getSessionId() },
  });
  return handleResponse<T>(res);
}

export async function apiPost<T = any>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Session-ID': getSessionId(),
    },
    body: body === undefined ? '{}' : JSON.stringify(body),
  });
  return handleResponse<T>(res);
}

export async function apiDelete<T = any>(path: string): Promise<T> {
  const res = await fetch(path, {
    method: 'DELETE',
    headers: { 'X-Session-ID': getSessionId() },
  });
  return handleResponse<T>(res);
}

export async function apiUpload<T = any>(path: string, formData: FormData): Promise<T> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'X-Session-ID': getSessionId() },
    body: formData,
  });
  return handleResponse<T>(res);
}

export async function apiDownload(
  path: string,
  filename: string,
  method: 'GET' | 'POST' = 'GET',
  formData?: FormData,
): Promise<void> {
  const res = await fetch(path, {
    method,
    headers: { 'X-Session-ID': getSessionId() },
    body: formData,
  });
  if (!res.ok) {
    const rawText = await res.text();
    throw new Error(extractErrorMessage(rawText, res.status, res.statusText));
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

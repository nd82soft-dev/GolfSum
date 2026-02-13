const crypto = require('crypto');

const RATE_LIMIT_WINDOW_MS = 10 * 60 * 1000;
const RATE_LIMIT_MAX = 30;

function getClientIp(req) {
  const xff = req.headers['x-forwarded-for'];
  if (typeof xff === 'string' && xff.length > 0) return xff.split(',')[0].trim();
  return req.socket?.remoteAddress || 'unknown';
}

function getRateLimitStore() {
  if (!globalThis.__gsRateLimit) {
    globalThis.__gsRateLimit = new Map();
  }
  return globalThis.__gsRateLimit;
}

function checkRateLimit(ip) {
  const now = Date.now();
  const store = getRateLimitStore();
  const entry = store.get(ip);
  if (!entry || now - entry.start > RATE_LIMIT_WINDOW_MS) {
    store.set(ip, { start: now, count: 1 });
    return { allowed: true, remaining: RATE_LIMIT_MAX - 1 };
  }
  if (entry.count >= RATE_LIMIT_MAX) {
    return { allowed: false, remaining: 0, resetMs: RATE_LIMIT_WINDOW_MS - (now - entry.start) };
  }
  entry.count += 1;
  return { allowed: true, remaining: RATE_LIMIT_MAX - entry.count };
}

function verifySignature(req) {
  const secret = process.env.GS_API_SIGNING_SECRET;
  if (!secret) return true;

  const timestamp = req.headers['x-gs-timestamp'];
  const signature = req.headers['x-gs-signature'];
  if (!timestamp || !signature) return false;

  const ts = Number(timestamp);
  if (!Number.isFinite(ts)) return false;

  const maxSkewMs = 5 * 60 * 1000;
  if (Math.abs(Date.now() - ts) > maxSkewMs) return false;

  const url = new URL(req.url, 'http://localhost');
  const mode = url.searchParams.get('mode') || '';
  const payload = `${timestamp}.${req.method}.${url.pathname}.${mode}`;
  const expected = crypto.createHmac('sha256', secret).update(payload).digest('hex');

  try {
    return crypto.timingSafeEqual(Buffer.from(expected), Buffer.from(signature));
  } catch {
    return false;
  }
}

async function readRawBody(req) {
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  return Buffer.concat(chunks);
}

module.exports = async (req, res) => {
  if (req.method !== 'POST') {
    res.statusCode = 405;
    res.setHeader('Allow', 'POST');
    res.end('Method Not Allowed');
    return;
  }

  const ip = getClientIp(req);
  const rate = checkRateLimit(ip);
  res.setHeader('X-RateLimit-Limit', RATE_LIMIT_MAX.toString());
  res.setHeader('X-RateLimit-Remaining', String(rate.remaining));
  if (!rate.allowed) {
    res.statusCode = 429;
    res.end('Rate limit exceeded. Please retry later.');
    return;
  }

  if (!verifySignature(req)) {
    res.statusCode = 401;
    res.end('Invalid request signature');
    return;
  }

  const backendBase = process.env.OCR_BACKEND_URL || '';
  if (!backendBase) {
    res.statusCode = 500;
    res.end('OCR backend not configured');
    return;
  }

  const upstreamBase = backendBase.replace(/\/+$/, '');
  const upstreamUrl = new URL(req.url, 'http://localhost');
  const mode = upstreamUrl.searchParams.get('mode');
  const targetUrl = `${upstreamBase}/scorecard/parse${mode ? `?mode=${encodeURIComponent(mode)}` : ''}`;

  const body = await readRawBody(req);
  const contentType = req.headers['content-type'] || 'application/octet-stream';

  const ocrApiKey = process.env.OCR_API_KEY || '';

  let upstreamResponse;
  try {
    upstreamResponse = await fetch(targetUrl, {
      method: 'POST',
      headers: {
        'content-type': contentType,
        ...(ocrApiKey ? { 'X-API-Key': ocrApiKey } : {}),
      },
      body
    });
  } catch (error) {
    res.statusCode = 502;
    res.end('Failed to reach OCR backend');
    return;
  }

  res.statusCode = upstreamResponse.status;
  const responseContentType = upstreamResponse.headers.get('content-type');
  if (responseContentType) res.setHeader('content-type', responseContentType);
  const responseBody = Buffer.from(await upstreamResponse.arrayBuffer());
  res.end(responseBody);
};

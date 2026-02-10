import { useState, useEffect, useCallback, useMemo } from "react";

// ─── Course Photos (originals by ND) ───
const PHOTOS = {
  hero: "/images/hero-vineyard-valley.jpg",
  foggy: "/images/foggy-morning.jpg",
  bunkers: "/images/elevated-tee-bunkers.jpg",
  uphill: "/images/uphill-fairway.jpg",
  overcast: "/images/overcast-hillside.jpg",
  panoramic: "/images/panoramic-clubhouse.jpg",
  pebble: "/images/pebble-beach-coast.jpg",
  hmb: "/images/hero-halfmoon-bay.jpg",
  hmb2: "/images/halfmoon-overcast.jpg",
};

// ─── Firebase Config ───
const FIREBASE_CONFIG = {
  apiKey: "AIzaSyDcHe0-Ii-sqbRqT8pLqgoP0gba9aoY9lQ",
  authDomain: "golfsum-a970f.firebaseapp.com",
  projectId: "golfsum-a970f",
};
const FIRESTORE_BASE = `https://firestore.googleapis.com/v1/projects/${FIREBASE_CONFIG.projectId}/databases/(default)/documents`;

// ─── Firestore Helpers ───
const fromFirestoreValue = (v) => {
  if (!v) return null;
  if ("stringValue" in v) return v.stringValue;
  if ("integerValue" in v) return Number(v.integerValue);
  if ("doubleValue" in v) return v.doubleValue;
  if ("booleanValue" in v) return v.booleanValue;
  if ("timestampValue" in v) return v.timestampValue;
  if ("mapValue" in v) return fromFirestoreFields(v.mapValue.fields || {});
  if ("arrayValue" in v) return (v.arrayValue.values || []).map(fromFirestoreValue);
  if ("nullValue" in v) return null;
  return null;
};
const fromFirestoreFields = (fields) => {
  const obj = {};
  for (const [k, v] of Object.entries(fields)) obj[k] = fromFirestoreValue(v);
  return obj;
};
const toFirestoreValue = (v) => {
  if (v === null || v === undefined) return { nullValue: null };
  if (Array.isArray(v)) return { arrayValue: { values: v.map(toFirestoreValue) } };
  if (v instanceof Date) return { timestampValue: v.toISOString() };
  if (typeof v === "number") return Number.isInteger(v) ? { integerValue: String(v) } : { doubleValue: v };
  if (typeof v === "boolean") return { booleanValue: v };
  if (typeof v === "object") return { mapValue: { fields: toFirestoreFields(v) } };
  return { stringValue: String(v) };
};
const toFirestoreFields = (obj) => {
  const fields = {};
  for (const [k, v] of Object.entries(obj || {})) fields[k] = toFirestoreValue(v);
  return fields;
};

// ─── Auth ───
const signInWithEmail = async (email, password) => {
  const r = await fetch(`https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=${FIREBASE_CONFIG.apiKey}`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, returnSecureToken: true }),
  });
  if (!r.ok) { const e = await r.json(); throw new Error(e.error?.message || "Sign in failed"); }
  return r.json();
};
const signUpWithEmail = async (email, password) => {
  const r = await fetch(`https://identitytoolkit.googleapis.com/v1/accounts:signUp?key=${FIREBASE_CONFIG.apiKey}`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password, returnSecureToken: true }),
  });
  if (!r.ok) { const e = await r.json(); throw new Error(e.error?.message || "Sign up failed"); }
  return r.json();
};

// ─── Firestore Queries ───
const firestoreList = async (path, token, pageSize = 100) => {
  const r = await fetch(`${FIRESTORE_BASE}/${path}?pageSize=${pageSize}`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!r.ok) return [];
  const d = await r.json();
  return (d.documents || []).map((doc) => ({ id: doc.name.split("/").pop(), ...fromFirestoreFields(doc.fields || {}) }));
};
const firestoreQuery = async (structuredQuery, token) => {
  const r = await fetch(`https://firestore.googleapis.com/v1/projects/${FIREBASE_CONFIG.projectId}/databases/(default)/documents:runQuery`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}) },
    body: JSON.stringify({ structuredQuery }),
  });
  if (!r.ok) return [];
  const d = await r.json();
  return d.map((row) => row.document).filter(Boolean).map((doc) => ({ id: doc.name.split("/").pop(), ...fromFirestoreFields(doc.fields || {}) }));
};
const firestoreAggregateCount = async (structuredQuery, token, alias = "count") => {
  const r = await fetch(`https://firestore.googleapis.com/v1/projects/${FIREBASE_CONFIG.projectId}/databases/(default)/documents:runAggregationQuery`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}) },
    body: JSON.stringify({
      structuredAggregationQuery: {
        structuredQuery,
        aggregations: [{ alias, count: {} }],
      },
    }),
  });
  if (!r.ok) return null;
  const d = await r.json();
  const v = d?.[0]?.result?.aggregateFields?.[alias];
  if (!v) return null;
  return v.integerValue != null ? Number(v.integerValue) : v.doubleValue ?? null;
};
const firestorePatch = async (path, token, updates) => {
  const params = Object.keys(updates || {}).map((k) => `updateMask.fieldPaths=${encodeURIComponent(k)}`).join("&");
  const url = `${FIRESTORE_BASE}/${path}${params ? `?${params}` : ""}`;
  const r = await fetch(url, {
    method: "PATCH",
    headers: { "Content-Type": "application/json", ...(token ? { Authorization: `Bearer ${token}` } : {}) },
    body: JSON.stringify({ fields: toFirestoreFields(updates) }),
  });
  return r.ok;
};
const listAllUsers = async (token) => {
  const r = await fetch(`${FIRESTORE_BASE}/users?pageSize=300`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!r.ok) return [];
  const d = await r.json();
  return (d.documents || []).map((doc) => ({ uid: doc.name.split("/").pop(), ...fromFirestoreFields(doc.fields || {}) }));
};
const getUserRounds = async (uid, token) => firestoreList(`users/${uid}/rounds`, token, 300);
const updateLastLogin = async (uid, token) => {
  const device = {
    userAgent: typeof navigator !== "undefined" ? navigator.userAgent : "",
    platform: typeof navigator !== "undefined" ? navigator.platform : "",
  };
  const updates = { lastLoginAt: new Date().toISOString(), lastLoginDevice: device };
  await firestorePatch(`users/${uid}`, token, updates);
};

// ─── Config ───
const ADMIN_EMAILS = ["support@golfsum.com"]; // ← Add your email

// ─── Theme ───
const C = {
  bg: "#0B0F13", bgCard: "#111820", bgElevated: "#1A2230", bgHover: "#1E2A3A",
  brand: "#10B981", brandDim: "rgba(16,185,129,0.12)", brandGlow: "rgba(16,185,129,0.25)",
  text: "#E8ECF0", textMuted: "#8B9CB5", textDim: "#5A6A80",
  border: "#1E2A38", borderLight: "#2A3848",
  red: "#EF4444", amber: "#F59E0B", blue: "#3B82F6",
};

// ─── Icons ───
const Icon = ({ name, size = 20, color = C.textMuted }) => {
  const s = { width: size, height: size, display: "inline-block", verticalAlign: "middle" };
  const paths = {
    golf: <svg style={s} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round"><path d="M12 18V3l7 4-7 4"/><path d="M9 17c-2.2.5-4 1.3-4 2.5C5 21 8.1 22 12 22s7-1 7-2.5c0-1.2-1.8-2-4-2.5"/></svg>,
    chart: <svg style={s} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round"><path d="M18 20V10M12 20V4M6 20v-6"/></svg>,
    users: <svg style={s} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round"><path d="M16 21v-2a4 4 0 00-4-4H6a4 4 0 00-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 00-3-3.87M16 3.13a4 4 0 010 7.75"/></svg>,
    target: <svg style={s} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>,
    camera: <svg style={s} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round"><path d="M23 19a2 2 0 01-2 2H3a2 2 0 01-2-2V8a2 2 0 012-2h4l2-3h6l2 3h4a2 2 0 012 2z"/><circle cx="12" cy="13" r="4"/></svg>,
    brain: <svg style={s} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round"><path d="M9.5 2A5.5 5.5 0 005 7.5c0 1 .3 2 .8 2.8A5.5 5.5 0 003 15c0 3 2.5 5.5 5.5 5.5h1M14.5 2A5.5 5.5 0 0120 7.5c0 1-.3 2-.8 2.8A5.5 5.5 0 0121 15c0 3-2.5 5.5-5.5 5.5h-1M12 2v20"/></svg>,
    arrow: <svg style={s} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round"><path d="M5 12h14M12 5l7 7-7 7"/></svg>,
    search: <svg style={s} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round"><circle cx="11" cy="11" r="8"/><path d="M21 21l-4.35-4.35"/></svg>,
    logout: <svg style={s} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round"><path d="M9 21H5a2 2 0 01-2-2V5a2 2 0 012-2h4M16 17l5-5-5-5M21 12H9"/></svg>,
    tool: <svg style={s} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round"><path d="M14.7 6.3a1 1 0 000 1.4l1.6 1.6a1 1 0 001.4 0l3.77-3.77a6 6 0 01-7.94 7.94l-6.91 6.91a2.12 2.12 0 01-3-3l6.91-6.91a6 6 0 017.94-7.94l-3.76 3.76z"/></svg>,
    flag: <svg style={s} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round"><path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1zM4 22v-7"/></svg>,
    download: <svg style={s} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/></svg>,
    check: <svg style={s} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round"><path d="M20 6L9 17l-5-5"/></svg>,
    home: <svg style={s} viewBox="0 0 24 24" fill="none" stroke={color} strokeWidth="2" strokeLinecap="round"><path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z"/><path d="M9 22V12h6v10"/></svg>,
  };
  return paths[name] || null;
};

// ─── Utility ───
const fmt = (n, d = 1) => n != null ? Number(n).toFixed(d) : "—";
const pct = (hit, total) => total > 0 ? Math.round((hit / total) * 100) + "%" : "—";
const fmtDate = (d) => {
  if (!d) return "—";
  const dt = typeof d === "string" ? new Date(d) : d instanceof Date ? d : new Date();
  return dt.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" });
};
const daysSince = (d) => {
  if (!d) return 999;
  const dt = typeof d === "string" ? new Date(d) : d;
  return Math.floor((Date.now() - dt.getTime()) / 86400000);
};

// ─── Photo gallery data ───
const GALLERY = [
  { src: PHOTOS.hero, alt: "Vineyard valley fairways", span: "wide" },
  { src: PHOTOS.pebble, alt: "Pebble Beach coastline", span: "normal" },
  { src: PHOTOS.hmb, alt: "Half Moon Bay oceanside", span: "normal" },
  { src: PHOTOS.panoramic, alt: "Panoramic hillside course", span: "wide" },
  { src: PHOTOS.bunkers, alt: "Elevated tee with bunkers", span: "normal" },
  { src: PHOTOS.foggy, alt: "Foggy morning tee box", span: "normal" },
  { src: PHOTOS.uphill, alt: "Rolling fairway", span: "normal" },
  { src: PHOTOS.overcast, alt: "Hill country golf", span: "normal" },
  { src: PHOTOS.hmb2, alt: "Cliffside resort course", span: "normal" },
];

// ─── Styles ───
const css = `
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=Instrument+Serif:ital@0;1&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body { font-family: 'DM Sans', sans-serif; background: ${C.bg}; color: ${C.text}; line-height: 1.6; -webkit-font-smoothing: antialiased; }
::selection { background: ${C.brandGlow}; color: ${C.text}; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: ${C.bg}; }
::-webkit-scrollbar-thumb { background: ${C.borderLight}; border-radius: 3px; }

@keyframes fadeUp { from { opacity: 0; transform: translateY(24px); } to { opacity: 1; transform: translateY(0); } }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes shimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
@keyframes kenburns { 0% { transform: scale(1.05); } 100% { transform: scale(1.15); } }

.fade-up { animation: fadeUp 0.6s ease-out both; }
.fade-in { animation: fadeIn 0.4s ease-out both; }
.stagger-1 { animation-delay: 0.1s; }
.stagger-2 { animation-delay: 0.2s; }
.stagger-3 { animation-delay: 0.3s; }
.stagger-4 { animation-delay: 0.4s; }
.stagger-5 { animation-delay: 0.5s; }
.serif { font-family: 'Instrument Serif', Georgia, serif; }

a { color: ${C.brand}; text-decoration: none; transition: color 0.2s; }
a:hover { color: #34D399; }

.btn {
  display: inline-flex; align-items: center; gap: 8px; padding: 12px 24px;
  border-radius: 10px; font-weight: 600; font-size: 15px; border: none;
  cursor: pointer; transition: all 0.2s; font-family: inherit;
}
.btn-primary { background: ${C.brand}; color: #fff; }
.btn-primary:hover { background: #0DA874; transform: translateY(-1px); box-shadow: 0 4px 20px ${C.brandGlow}; }
.btn-secondary { background: ${C.bgElevated}; color: ${C.text}; border: 1px solid ${C.border}; }
.btn-secondary:hover { background: ${C.bgHover}; border-color: ${C.borderLight}; }
.btn-ghost { background: transparent; color: ${C.textMuted}; padding: 8px 16px; }
.btn-ghost:hover { color: ${C.text}; background: ${C.bgElevated}; }
.btn-sm { padding: 8px 16px; font-size: 13px; border-radius: 8px; }

.card {
  background: ${C.bgCard}; border: 1px solid ${C.border}; border-radius: 16px;
  padding: 24px; transition: border-color 0.2s;
}
.card:hover { border-color: ${C.borderLight}; }

.input {
  width: 100%; padding: 12px 16px; background: ${C.bgElevated}; border: 1px solid ${C.border};
  border-radius: 10px; color: ${C.text}; font-size: 15px; font-family: inherit;
  transition: border-color 0.2s; outline: none;
}
.input:focus { border-color: ${C.brand}; box-shadow: 0 0 0 3px ${C.brandDim}; }
.input::placeholder { color: ${C.textDim}; }

.badge { display: inline-flex; align-items: center; gap: 4px; padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; }
.badge-green { background: ${C.brandDim}; color: ${C.brand}; }
.badge-red { background: rgba(239,68,68,0.12); color: ${C.red}; }
.badge-amber { background: rgba(245,158,11,0.12); color: ${C.amber}; }
.badge-blue { background: rgba(59,130,246,0.12); color: ${C.blue}; }

.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 12px; }
.stat-box { background: ${C.bgElevated}; border-radius: 12px; padding: 16px; text-align: center; border: 1px solid ${C.border}; }
.stat-value { font-size: 28px; font-weight: 700; color: ${C.text}; line-height: 1.2; }
.stat-label { font-size: 12px; color: ${C.textMuted}; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px; }

.table { width: 100%; border-collapse: collapse; }
.table th { text-align: left; padding: 10px 12px; font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em; color: ${C.textDim}; border-bottom: 1px solid ${C.border}; font-weight: 600; }
.table td { padding: 12px; border-bottom: 1px solid ${C.border}; font-size: 14px; color: ${C.textMuted}; }
.table tr:hover td { background: ${C.bgElevated}; }

.loading-bar { height: 3px; background: linear-gradient(90deg, transparent, ${C.brand}, transparent); background-size: 200% 100%; animation: shimmer 1.5s infinite; border-radius: 2px; }

.photo-grid {
  display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;
}
.photo-grid .wide { grid-column: span 2; }
.photo-grid img {
  width: 100%; height: 200px; object-fit: cover; border-radius: 12px;
  transition: transform 0.4s, filter 0.4s; filter: brightness(0.85);
}
.photo-grid img:hover { transform: scale(1.02); filter: brightness(1); }

@media (max-width: 768px) {
  .photo-grid { grid-template-columns: repeat(2, 1fr); }
  .photo-grid .wide { grid-column: span 2; }
  .photo-grid img { height: 140px; }
}
`;

// ─── Premium feature list ───
const premiumFeatureList = [
  "Everything in Free, plus:", "FIR & GIR with miss direction", "Scrambling & short game",
  "Performance insights", "Scoring trends & averages", "Full OCR stat extraction",
  "Approach distance zones", "Club tracking", "Weather integration",
  "Data export (CSV/PDF)", "Shot pattern heatmaps", "Practice plan generation",
];

// ─── App ───
export default function GolfSumSite() {
  const [page, setPage] = useState("home");
  const [user, setUser] = useState(null);
  const [isAdmin, setIsAdmin] = useState(false);

  useEffect(() => {
    try {
      const s = sessionStorage?.getItem?.("gs_session");
      if (s) { const p = JSON.parse(s); setUser(p); setIsAdmin(ADMIN_EMAILS.includes(p.email?.toLowerCase())); }
    } catch {}
  }, []);

  const handleLogin = (u) => {
    setUser(u); setIsAdmin(ADMIN_EMAILS.includes(u.email?.toLowerCase()));
    try { sessionStorage?.setItem?.("gs_session", JSON.stringify(u)); } catch {}
    setPage("dashboard");
  };
  const handleLogout = () => {
    setUser(null); setIsAdmin(false);
    try { sessionStorage?.removeItem?.("gs_session"); } catch {}
    setPage("home");
  };

  return (
    <div style={{ minHeight: "100vh" }}>
      <style>{css}</style>
      <Nav page={page} nav={setPage} user={user} isAdmin={isAdmin} onLogout={handleLogout} />
      {page === "home" && <HomePage nav={setPage} />}
      {page === "features" && <FeaturesPage />}
      {page === "pricing" && <PricingPage />}
      {page === "privacy" && <PrivacyPage />}
      {page === "terms" && <TermsPage />}
      {page === "login" && <LoginPage onLogin={handleLogin} />}
      {page === "dashboard" && user && <DashboardPage user={user} />}
      {page === "admin" && user && isAdmin && <AdminPage user={user} />}
      <Footer nav={setPage} />
    </div>
  );
}

// ─── Nav ───
function Nav({ page, nav, user, isAdmin, onLogout }) {
  return (
    <nav style={{ position: "sticky", top: 0, zIndex: 100, background: "rgba(11,15,19,0.85)", backdropFilter: "blur(20px)", borderBottom: `1px solid ${C.border}` }}>
      <div style={{ maxWidth: 1200, margin: "0 auto", padding: "0 24px", display: "flex", alignItems: "center", justifyContent: "space-between", height: 64 }}>
        <button onClick={() => nav("home")} style={{ background: "none", border: "none", cursor: "pointer", display: "flex", alignItems: "center", gap: 10 }}>
          <div style={{ padding: "6px 8px", borderRadius: 10, background: "rgba(11,15,19,0.6)", border: `1px solid ${C.border}` }}>
            <img src="/images/golfsum-logo.png" alt="GolfSum" style={{ height: 28, width: "auto", display: "block", filter: "drop-shadow(0 2px 6px rgba(0,0,0,0.6)) brightness(1.1)" }} />
          </div>
        </button>
        <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
          {["features", "pricing"].map((p) => (
            <button key={p} onClick={() => nav(p)} className="btn-ghost" style={{ color: page === p ? C.text : C.textMuted, fontWeight: page === p ? 600 : 400, textTransform: "capitalize" }}>{p}</button>
          ))}
          {user ? (
            <>
              <button onClick={() => nav("dashboard")} className="btn-ghost" style={{ color: page === "dashboard" ? C.text : C.textMuted, fontWeight: page === "dashboard" ? 600 : 400 }}>Dashboard</button>
              {isAdmin && <button onClick={() => nav("admin")} className="btn-ghost" style={{ color: page === "admin" ? C.amber : C.textMuted }}>Admin</button>}
              <button onClick={onLogout} className="btn-ghost" title="Log out"><Icon name="logout" size={18} /></button>
            </>
          ) : (
            <button onClick={() => nav("login")} className="btn btn-primary btn-sm" style={{ marginLeft: 8 }}>Sign In</button>
          )}
        </div>
      </div>
    </nav>
  );
}

// ─── Home ───
function HomePage({ nav }) {
  return (
    <div>
      {/* Hero with photo background */}
      <section style={{ position: "relative", overflow: "hidden", minHeight: 540 }}>
        <div style={{
          position: "absolute", inset: 0, backgroundImage: `url(${PHOTOS.hero})`,
          backgroundSize: "cover", backgroundPosition: "center 40%",
          animation: "kenburns 20s ease-in-out infinite alternate",
        }} />
        <div style={{ position: "absolute", inset: 0, background: "linear-gradient(to bottom, rgba(11,15,19,0.55) 0%, rgba(11,15,19,0.75) 60%, rgba(11,15,19,0.95) 100%)" }} />
        <div className="fade-up" style={{ position: "relative", zIndex: 1, maxWidth: 700, margin: "0 auto", textAlign: "center", padding: "120px 24px 100px" }}>
          <div className="badge badge-green" style={{ marginBottom: 20 }}>Now in Beta · iOS & Android</div>
          <h1 className="serif" style={{ fontSize: "clamp(36px, 5.5vw, 64px)", fontWeight: 400, lineHeight: 1.1, marginBottom: 20, letterSpacing: "-0.02em", textShadow: "0 2px 20px rgba(0,0,0,0.5)" }}>
            Track. Analyze. <span style={{ color: C.brand }}>Improve.</span>
          </h1>
          <p style={{ fontSize: 18, color: "#C8D0DC", maxWidth: 500, margin: "0 auto 36px", lineHeight: 1.7, textShadow: "0 1px 8px rgba(0,0,0,0.4)" }}>
            GolfSum turns your scorecards into coaching insights. No GPS sensors. No swing cameras. Just data-driven improvement from the rounds you already play.
          </p>
          <div style={{ display: "flex", gap: 12, justifyContent: "center", flexWrap: "wrap" }}>
            <button className="btn btn-primary" style={{ fontSize: 16, padding: "14px 32px" }} onClick={() => nav("pricing")}>
              Get Started <Icon name="arrow" size={18} color="#fff" />
            </button>
            <button className="btn btn-secondary" onClick={() => nav("features")}>See Features</button>
          </div>
        </div>
      </section>

      {/* Stats Bar */}
      <section style={{ borderTop: `1px solid ${C.border}`, borderBottom: `1px solid ${C.border}`, padding: "32px 24px" }}>
        <div style={{ maxWidth: 800, margin: "0 auto", display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 24, textAlign: "center" }}>
          {[["$5.99/mo", "or $49.99/yr (save 31%)"], ["0", "Hardware Required"], ["WHS", "Compliant"]].map(([val, label], i) => (
            <div key={i} className={`fade-up stagger-${i + 1}`}>
              <div style={{ fontSize: 28, fontWeight: 700, color: C.text }}>{val}</div>
              <div style={{ fontSize: 13, color: C.textMuted }}>{label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Feature Cards with photos */}
      <section style={{ padding: "80px 24px", maxWidth: 1100, margin: "0 auto" }}>
        <h2 className="serif fade-up" style={{ fontSize: 36, fontWeight: 400, textAlign: "center", marginBottom: 48 }}>
          Everything you need to <span style={{ fontStyle: "italic", color: C.brand }}>lower your scores</span>
        </h2>
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: 20 }}>
          {[
            { icon: "chart", title: "Deep Stat Tracking", desc: "FIR, GIR, scrambling, putts, approach distances, penalties, all from manual entry or OCR scorecard import.", photo: PHOTOS.bunkers },
            { icon: "camera", title: "OCR Scorecard Import", desc: "Photograph old scorecards. Automatic extraction pulls course details, yardages, pars, and your scores.", photo: PHOTOS.uphill },
            { icon: "brain", title: "Performance Insights", desc: "Personalized tips calibrated to your handicap tier. Not generic, based on your patterns across rounds.", photo: PHOTOS.panoramic },
            { icon: "target", title: "WHS Handicap Index", desc: "Full World Handicap System compliance with Net Double Bogey, 9-hole pairing, slope adjustments, and more.", photo: PHOTOS.overcast },
            { icon: "chart", title: "Scoring Trends", desc: "Track how your game evolves. Front 9 vs Back 9, par type breakdowns, weather impact splits.", photo: PHOTOS.foggy },
            { icon: "download", title: "Data Export", desc: "Export your stats as CSV or share formatted round summaries. Your data is yours.", photo: PHOTOS.hmb2 },
          ].map((f, i) => (
            <div key={i} className={`card fade-up stagger-${(i % 3) + 1}`} style={{ padding: 0, overflow: "hidden" }}>
              <div style={{ height: 140, backgroundImage: `url(${f.photo})`, backgroundSize: "cover", backgroundPosition: "center", position: "relative" }}>
                <div style={{ position: "absolute", inset: 0, background: "linear-gradient(to bottom, transparent 30%, rgba(17,24,32,0.95) 100%)" }} />
              </div>
              <div style={{ padding: "16px 20px 20px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                  <div style={{ width: 32, height: 32, borderRadius: 8, background: C.brandDim, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0 }}>
                    <Icon name={f.icon} size={16} color={C.brand} />
                  </div>
                  <h3 style={{ fontSize: 16, fontWeight: 600 }}>{f.title}</h3>
                </div>
                <p style={{ fontSize: 14, color: C.textMuted, lineHeight: 1.6 }}>{f.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Photo Gallery */}
      <section style={{ padding: "0 24px 60px", maxWidth: 1100, margin: "0 auto" }}>
        <h2 className="serif fade-up" style={{ fontSize: 28, fontWeight: 400, textAlign: "center", marginBottom: 8 }}>
          From the <span style={{ fontStyle: "italic" }}>courses we love</span>
        </h2>
        <p className="fade-up stagger-1" style={{ textAlign: "center", color: C.textDim, marginBottom: 32, fontSize: 14 }}>Shot on location by the GolfSum team</p>
        <div className="photo-grid fade-up stagger-2">
          {GALLERY.map((p, i) => (
            <div key={i} className={p.span === "wide" ? "wide" : ""}>
              <img src={p.src} alt={p.alt} loading="lazy" />
            </div>
          ))}
        </div>
      </section>

      {/* CTA with photo */}
      <section style={{ padding: "60px 24px 100px", textAlign: "center" }}>
        <div className="fade-up" style={{
          maxWidth: 700, margin: "0 auto", borderRadius: 20, overflow: "hidden", position: "relative",
        }}>
          <div style={{
            position: "absolute", inset: 0, backgroundImage: `url(${PHOTOS.hmb})`,
            backgroundSize: "cover", backgroundPosition: "center",
          }} />
          <div style={{ position: "absolute", inset: 0, background: "rgba(11,15,19,0.8)", backdropFilter: "blur(2px)" }} />
          <div style={{ position: "relative", padding: "56px 32px" }}>
            <h2 className="serif" style={{ fontSize: 30, fontWeight: 400, marginBottom: 12 }}>Ready to play smarter?</h2>
            <p style={{ color: C.textMuted, marginBottom: 28 }}>Start with 3 rounds. Let the data show you where to improve.</p>
            <button className="btn btn-primary" style={{ fontSize: 16, padding: "14px 32px" }} onClick={() => nav("pricing")}>
              Start Free Rounds <Icon name="arrow" size={18} color="#fff" />
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}

// ─── Features ───
function FeaturesPage() {
  const features = [
    { icon: "chart", title: "Manual Score Entry", items: ["Hole-by-hole score and putts (free)", "FIR/GIR with miss direction arrows", "Up & Down tracking", "Approach distance zones", "Club selection per shot", "Penalty strokes & bunkers", "Basic and Advanced scoring modes"] },
    { icon: "camera", title: "OCR Scorecard Import", items: ["Photograph any printed scorecard", "Auto-extracts course layout, yardages, pars", "Auto-detects tee boxes, ratings, slopes", "Player score extraction (Premium)", "Editable review screen before save", "Course saved to shared catalog"] },
    { icon: "brain", title: "Performance Insights", items: ["Handicap-tiered analysis (LOW/MID/HIGH)", "Miss direction pattern detection", "Bogey train / momentum analysis", "Scramble quality breakdown", "Approach distance zone efficiency", "Practice plan generation", "Three-putt rate tracking"] },
    { icon: "target", title: "WHS Handicap", items: ["Full World Handicap System compliance", "Net Double Bogey adjustment", "9-hole round pairing", "Incomplete round handling (10+ holes)", "Fallback table for < 20 rounds", "Course rating & slope integration", "Differential calculation"] },
    { icon: "flag", title: "Round Detail & History", items: ["Score color-coded scorecard", "Scoring distribution visualization", "Personal best detection", "Round comparison tools", "Round rating & notes", "Weather conditions integration", "Front 9 vs Back 9 splits"] },
  ];
  return (
    <section style={{ padding: "80px 24px", maxWidth: 900, margin: "0 auto" }}>
      {/* Hero banner */}
      <div className="fade-up" style={{ borderRadius: 16, overflow: "hidden", marginBottom: 48, position: "relative", height: 200 }}>
        <img src={PHOTOS.panoramic} alt="" style={{ width: "100%", height: "100%", objectFit: "cover", filter: "brightness(0.6)" }} />
        <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center" }}>
          <h1 className="serif" style={{ fontSize: 42, fontWeight: 400, textShadow: "0 2px 16px rgba(0,0,0,0.6)" }}>Features</h1>
          <p style={{ color: "#C8D0DC", fontSize: 17 }}>Everything under the hood</p>
        </div>
      </div>
      {features.map((f, i) => (
        <div key={i} className={`card fade-up stagger-${(i % 3) + 1}`} style={{ marginBottom: 20, display: "flex", gap: 24, alignItems: "flex-start" }}>
          <div style={{ width: 44, height: 44, borderRadius: 12, background: C.brandDim, display: "flex", alignItems: "center", justifyContent: "center", flexShrink: 0, marginTop: 2 }}>
            <Icon name={f.icon} size={22} color={C.brand} />
          </div>
          <div>
            <h3 style={{ fontSize: 18, fontWeight: 600, marginBottom: 12 }}>{f.title}</h3>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "6px 24px" }}>
              {f.items.map((item, j) => (
                <div key={j} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 14, color: C.textMuted }}>
                  <Icon name="check" size={14} color={C.brand} /> {item}
                </div>
              ))}
            </div>
          </div>
        </div>
      ))}
    </section>
  );
}

// ─── Pricing ───
function PricingPage() {
  return (
    <section style={{ padding: "80px 24px", maxWidth: 1000, margin: "0 auto" }}>
      <h1 className="serif fade-up" style={{ fontSize: 42, fontWeight: 400, textAlign: "center", marginBottom: 12 }}>Simple Pricing</h1>
      <p className="fade-up stagger-1" style={{ textAlign: "center", color: C.textMuted, marginBottom: 48, fontSize: 17 }}>Start free with 3 premium rounds. Upgrade when you're ready for the full picture.</p>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 20 }}>
        {/* Free */}
        <div className="card fade-up stagger-1" style={{ padding: 32 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: C.textMuted, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>Free</div>
          <div style={{ fontSize: 40, fontWeight: 700, marginBottom: 4 }}>$0</div>
          <div style={{ fontSize: 14, color: C.textDim, marginBottom: 24 }}>Forever</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {["Score & putts per hole", "Round history", "WHS Handicap Index", "OCR course import (yardages, pars, tees, ratings)", "Course search", "Manual course entry"].map((f, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 14, color: C.textMuted }}>
                <Icon name="check" size={14} color={C.brand} /> {f}
              </div>
            ))}
          </div>
        </div>
        {/* Monthly */}
        <div className="card fade-up stagger-2" style={{ padding: 32 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: C.brand, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>Premium Monthly</div>
          <div style={{ fontSize: 40, fontWeight: 700, marginBottom: 4 }}>$5.99<span style={{ fontSize: 16, fontWeight: 400, color: C.textMuted }}>/mo</span></div>
          <div style={{ fontSize: 14, color: C.textDim, marginBottom: 24 }}>Cancel anytime</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10, marginBottom: 28 }}>
            {premiumFeatureList.map((f, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 14, color: i === 0 ? C.text : C.textMuted, fontWeight: i === 0 ? 600 : 400 }}>
                <Icon name="check" size={14} color={C.brand} /> {f}
              </div>
            ))}
          </div>
          <button className="btn btn-secondary" style={{ width: "100%", justifyContent: "center" }}>Subscribe Monthly</button>
        </div>
        {/* Annual */}
        <div className="card fade-up stagger-3" style={{ padding: 32, borderColor: C.brand, position: "relative", background: `linear-gradient(135deg, ${C.bgCard}, rgba(16,185,129,0.04))` }}>
          <div className="badge badge-green" style={{ position: "absolute", top: 8, right: 12 }}>Save 31%</div>
          <div style={{ fontSize: 13, fontWeight: 600, color: C.brand, textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>Premium Annual</div>
          <div style={{ display: "flex", alignItems: "baseline", gap: 8, marginBottom: 4 }}>
            <span style={{ fontSize: 40, fontWeight: 700 }}>$49.99</span>
            <span style={{ fontSize: 16, color: C.textMuted }}>/year</span>
          </div>
          <div style={{ fontSize: 14, color: C.textDim, marginBottom: 4 }}>$4.17/month · 3 free premium rounds</div>
          <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 24 }}>
            <span style={{ fontSize: 14, color: C.textDim, textDecoration: "line-through" }}>$71.88/yr</span>
            <span style={{ fontSize: 14, color: C.brand, fontWeight: 600 }}>You save $21.89</span>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10, marginBottom: 28 }}>
            {premiumFeatureList.map((f, i) => (
              <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 14, color: i === 0 ? C.text : C.textMuted, fontWeight: i === 0 ? 600 : 400 }}>
                <Icon name="check" size={14} color={C.brand} /> {f}
              </div>
            ))}
          </div>
          <button className="btn btn-primary" style={{ width: "100%", justifyContent: "center" }}>Start Free Rounds</button>
        </div>
      </div>
    </section>
  );
}

// ─── Privacy ───
function PrivacyPage() {
  return (
    <section style={{ padding: "80px 24px", maxWidth: 740, margin: "0 auto" }}>
      <h1 className="serif fade-up" style={{ fontSize: 36, marginBottom: 8 }}>Privacy Policy</h1>
      <p className="fade-up stagger-1" style={{ color: C.textDim, marginBottom: 40, fontSize: 14 }}>Last updated: February 2026</p>
      {[
        { t: "1. Information We Collect", b: "We collect the information you provide when creating an account (email address) and the golf data you enter or import (scores, stats, course details). We also collect basic device and usage analytics to improve the app. We do not sell your personal data to third parties." },
        { t: "2. How We Use Your Data", b: "Your golf data is used to calculate your handicap index, generate coaching insights, and display your statistics. Account data is used for authentication and cloud sync. Anonymous, aggregated data may be used to improve our OCR accuracy and course catalog." },
        { t: "3. Data Storage & Security", b: "Your data is stored in Google Firebase (Firestore) with encryption at rest and in transit. Authentication is handled via Firebase Auth. We implement role-based access controls so users can only access their own data. Course catalog data contributed via OCR is shared across all users to improve the community database." },
        { t: "4. Third-Party Services", b: "We use Firebase (Google Cloud) for authentication, data storage, and hosting. We use RevenueCat for subscription management. We use Azure Document Intelligence for OCR processing. These services have their own privacy policies. We do not share your personal data with advertisers." },
        { t: "5. Your Rights", b: "You can export all your data at any time via the Data Export feature in the app. You can delete your account and all associated data by contacting support. We will respond to data deletion requests within 30 days." },
        { t: "6. Cookies & Tracking", b: "Our website uses essential cookies for authentication sessions. We do not use third-party advertising cookies or tracking pixels." },
        { t: "7. Children's Privacy", b: "GolfSum is not directed to children under 13. We do not knowingly collect data from children under 13. If you believe a child has provided us personal data, contact us and we will delete it." },
        { t: "8. Changes to This Policy", b: "We may update this policy from time to time. We will notify you of material changes via the app or email. Continued use after changes constitutes acceptance." },
        { t: "9. Contact", b: "For privacy questions or data requests, contact us at privacy@golfsum.com." },
      ].map((s, i) => (
        <div key={i} className={`fade-up stagger-${(i % 3) + 1}`} style={{ marginBottom: 28 }}>
          <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 8, color: C.text }}>{s.t}</h2>
          <p style={{ fontSize: 15, color: C.textMuted, lineHeight: 1.7 }}>{s.b}</p>
        </div>
      ))}
    </section>
  );
}

// ─── Terms ───
function TermsPage() {
  return (
    <section style={{ padding: "80px 24px", maxWidth: 740, margin: "0 auto" }}>
      <h1 className="serif fade-up" style={{ fontSize: 36, marginBottom: 8 }}>Terms of Service</h1>
      <p className="fade-up stagger-1" style={{ color: C.textDim, marginBottom: 40, fontSize: 14 }}>Last updated: February 2026</p>
      {[
        { t: "1. Acceptance of Terms", b: "By downloading, installing, or using GolfSum, you agree to these Terms of Service. If you do not agree, do not use the app." },
        { t: "2. Description of Service", b: "GolfSum is a golf analytics application that allows you to track scores, analyze statistics, and receive coaching insights. The service is provided on an 'as is' basis." },
        { t: "3. Accounts", b: "You are responsible for maintaining the security of your account credentials. You must provide accurate information when creating an account." },
        { t: "4. Subscriptions & Payments", b: "Premium features require an active subscription ($5.99/month or $49.99/year). Subscriptions are billed through Apple App Store or Google Play Store. Refunds are handled per the respective store's policy. We offer three free premium rounds; after that, you keep access to basic features unless you upgrade." },
        { t: "5. User Data & Content", b: "You retain ownership of all golf data you enter. By using the OCR import feature, you grant us a license to use extracted course data (yardages, pars, ratings) in our community course catalog." },
        { t: "6. Handicap Calculations", b: "GolfSum implements WHS calculations to the best of our ability. However, GolfSum is not an official handicap service and our calculations should not be used for official tournament purposes unless verified by an authorized handicap provider." },
        { t: "7. Limitation of Liability", b: "GolfSum is provided 'as is' without warranties of any kind. We are not liable for any indirect, incidental, or consequential damages arising from your use of the app. Our total liability is limited to the amount you paid in the prior 12 months." },
        { t: "8. Termination", b: "We may terminate or suspend your account for violation of these terms. You may cancel your subscription and delete your account at any time." },
        { t: "9. Changes to Terms", b: "We may update these terms from time to time. Material changes will be communicated via the app. Continued use constitutes acceptance." },
        { t: "10. Contact", b: "For questions about these terms, contact us at support@golfsum.com." },
      ].map((s, i) => (
        <div key={i} className={`fade-up stagger-${(i % 3) + 1}`} style={{ marginBottom: 28 }}>
          <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 8, color: C.text }}>{s.t}</h2>
          <p style={{ fontSize: 15, color: C.textMuted, lineHeight: 1.7 }}>{s.b}</p>
        </div>
      ))}
    </section>
  );
}

// ─── Login ───
function LoginPage({ onLogin }) {
  const [email, setEmail] = useState("");
  const [pass, setPass] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState("signin");

  const handleSubmit = async () => {
    setError(""); setLoading(true);
    try {
      const res = await (mode === "signin" ? signInWithEmail : signUpWithEmail)(email, pass);
      onLogin({ uid: res.localId, email: res.email, idToken: res.idToken, refreshToken: res.refreshToken });
      updateLastLogin(res.localId, res.idToken);
    } catch (e) { setError(e.message.replace(/_/g, " ").toLowerCase()); }
    setLoading(false);
  };

  return (
    <section style={{ padding: "100px 24px", maxWidth: 420, margin: "0 auto" }}>
      <div className="card fade-up" style={{ padding: 36 }}>
        <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 4, textAlign: "center" }}>{mode === "signin" ? "Welcome back" : "Create account"}</h2>
        <p style={{ fontSize: 14, color: C.textMuted, textAlign: "center", marginBottom: 28 }}>
          {mode === "signin" ? "Sign in to view your dashboard" : "Get started with GolfSum"}
        </p>
        {error && <div className="badge badge-red" style={{ marginBottom: 16, width: "100%", justifyContent: "center", padding: "8px 12px" }}>{error}</div>}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <input className="input" type="email" placeholder="Email" value={email} onChange={(e) => setEmail(e.target.value)} />
          <input className="input" type="password" placeholder="Password" value={pass} onChange={(e) => setPass(e.target.value)} onKeyDown={(e) => e.key === "Enter" && handleSubmit()} />
          <button className="btn btn-primary" style={{ width: "100%", justifyContent: "center", marginTop: 4 }} onClick={handleSubmit} disabled={loading}>
            {loading ? "Loading..." : mode === "signin" ? "Sign In" : "Create Account"}
          </button>
        </div>
        <p style={{ fontSize: 13, color: C.textDim, textAlign: "center", marginTop: 20 }}>
          {mode === "signin" ? "Don't have an account? " : "Already have an account? "}
          <a href="#" onClick={(e) => { e.preventDefault(); setMode(mode === "signin" ? "signup" : "signin"); setError(""); }}>
            {mode === "signin" ? "Sign up" : "Sign in"}
          </a>
        </p>
      </div>
    </section>
  );
}

// ─── User Dashboard ───
function DashboardPage({ user }) {
  const [rounds, setRounds] = useState([]);
  const [loading, setLoading] = useState(true);
  const [tab, setTab] = useState("overview");

  useEffect(() => {
    (async () => {
      setLoading(true);
      try {
        const data = await getUserRounds(user.uid, user.idToken);
        setRounds(data.sort((a, b) => new Date(b.date || 0) - new Date(a.date || 0)));
      } catch (e) { console.error(e); }
      setLoading(false);
    })();
  }, [user]);

  const stats = useMemo(() => {
    if (!rounds.length) return null;
    const scored = rounds.filter((r) => r.score > 0);
    const avgScore = scored.length ? scored.reduce((s, r) => s + r.score, 0) / scored.length : 0;
    const puttRounds = scored.filter((r) => r.stats?.putts);
    const avgPutts = puttRounds.length ? puttRounds.reduce((s, r) => s + r.stats.putts, 0) / puttRounds.length : 0;
    const firRounds = scored.filter((r) => r.stats?.fairwaysPossible > 0);
    const firPct = firRounds.length ? firRounds.reduce((s, r) => s + (r.stats.fairways || 0), 0) / firRounds.reduce((s, r) => s + r.stats.fairwaysPossible, 0) * 100 : null;
    const girRounds = scored.filter((r) => r.stats?.greensPossible > 0);
    const girPct = girRounds.length ? girRounds.reduce((s, r) => s + (r.stats.greens || 0), 0) / girRounds.reduce((s, r) => s + r.stats.greensPossible, 0) * 100 : null;
    const bestScore = scored.length ? Math.min(...scored.map((r) => r.score)) : null;
    const last5 = scored.slice(0, 5);
    const last5Avg = last5.length ? last5.reduce((s, r) => s + r.score, 0) / last5.length : null;
    return { total: scored.length, avgScore, avgPutts, firPct, girPct, bestScore, last5Avg };
  }, [rounds]);

  if (loading) return <section style={{ padding: "80px 24px", textAlign: "center" }}><div className="loading-bar" style={{ maxWidth: 200, margin: "40px auto" }} /><p style={{ color: C.textMuted }}>Loading your rounds...</p></section>;

  return (
    <section style={{ padding: "40px 24px 80px", maxWidth: 1100, margin: "0 auto" }}>
      <div className="fade-up" style={{ marginBottom: 32 }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 4 }}>Your Dashboard</h1>
        <p style={{ fontSize: 14, color: C.textMuted }}>{user.email} · {rounds.length} round{rounds.length !== 1 ? "s" : ""}</p>
      </div>
      {!rounds.length ? (
        <div className="card fade-up" style={{ textAlign: "center", padding: 48 }}>
          <Icon name="golf" size={40} color={C.textDim} />
          <h3 style={{ fontSize: 18, fontWeight: 600, marginTop: 16, marginBottom: 8 }}>No rounds yet</h3>
          <p style={{ color: C.textMuted, fontSize: 14 }}>Open the GolfSum app and track your first round to see your stats here.</p>
        </div>
      ) : (
        <>
          <div className="stat-grid fade-up stagger-1" style={{ marginBottom: 24 }}>
            <div className="stat-box"><div className="stat-value">{stats.total}</div><div className="stat-label">Rounds</div></div>
            <div className="stat-box"><div className="stat-value">{fmt(stats.avgScore, 1)}</div><div className="stat-label">Avg Score</div></div>
            <div className="stat-box"><div className="stat-value">{fmt(stats.avgPutts, 1)}</div><div className="stat-label">Avg Putts</div></div>
            <div className="stat-box"><div className="stat-value">{stats.bestScore || "—"}</div><div className="stat-label">Best Score</div></div>
            <div className="stat-box"><div className="stat-value">{stats.firPct != null ? fmt(stats.firPct, 0) + "%" : "—"}</div><div className="stat-label">FIR%</div></div>
            <div className="stat-box"><div className="stat-value">{stats.girPct != null ? fmt(stats.girPct, 0) + "%" : "—"}</div><div className="stat-label">GIR%</div></div>
            <div className="stat-box"><div className="stat-value">{stats.last5Avg ? fmt(stats.last5Avg, 1) : "—"}</div><div className="stat-label">Last 5 Avg</div></div>
            <div className="stat-box"><div className="stat-value" style={{ color: C.brand }}>—</div><div className="stat-label">Handicap Index</div></div>
          </div>
          <div style={{ display: "flex", gap: 4, marginBottom: 20, borderBottom: `1px solid ${C.border}` }}>
            {["overview", "history"].map((t) => (
              <button key={t} onClick={() => setTab(t)} style={{ background: "none", border: "none", padding: "10px 16px", cursor: "pointer", fontFamily: "inherit", fontSize: 14, fontWeight: tab === t ? 600 : 400, color: tab === t ? C.text : C.textMuted, borderBottom: tab === t ? `2px solid ${C.brand}` : "2px solid transparent", textTransform: "capitalize" }}>{t}</button>
            ))}
          </div>
          {tab === "overview" && (
            <div className="fade-in">
              <div className="card" style={{ marginBottom: 20, padding: 20 }}>
                <h3 style={{ fontSize: 15, fontWeight: 600, marginBottom: 16 }}>Scoring Trend</h3>
                <div style={{ display: "flex", alignItems: "flex-end", gap: 6, height: 100 }}>
                  {rounds.slice(0, 20).reverse().map((r, i) => {
                    const scores = rounds.slice(0, 20).map((x) => x.score);
                    const min = Math.min(...scores), max = Math.max(...scores), range = max - min || 1;
                    const h = ((r.score - min) / range) * 60 + 20;
                    return (<div key={i} style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
                      <div style={{ width: "100%", maxWidth: 28, height: h, borderRadius: 4, background: r.score <= (min + range * 0.3) ? C.brand : r.score >= (min + range * 0.7) ? "rgba(239,68,68,0.5)" : C.borderLight }} />
                      <span style={{ fontSize: 10, color: C.textDim }}>{r.score}</span>
                    </div>);
                  })}
                </div>
              </div>
              <div className="card" style={{ padding: 0, overflow: "hidden" }}>
                <div style={{ padding: "16px 20px", borderBottom: `1px solid ${C.border}` }}><h3 style={{ fontSize: 15, fontWeight: 600 }}>Recent Rounds</h3></div>
                <table className="table"><thead><tr><th>Date</th><th>Course</th><th>Score</th><th>Putts</th><th>FIR</th><th>GIR</th></tr></thead><tbody>
                  {rounds.slice(0, 10).map((r, i) => (
                    <tr key={i}><td>{fmtDate(r.date)}</td><td style={{ color: C.text, fontWeight: 500 }}>{r.courseName || "—"}</td><td style={{ fontWeight: 600, color: C.text }}>{r.score}</td><td>{r.stats?.putts || "—"}</td><td>{pct(r.stats?.fairways, r.stats?.fairwaysPossible)}</td><td>{pct(r.stats?.greens, r.stats?.greensPossible)}</td></tr>
                  ))}
                </tbody></table>
              </div>
            </div>
          )}
          {tab === "history" && (
            <div className="fade-in card" style={{ padding: 0, overflow: "auto" }}>
              <table className="table"><thead><tr><th>Date</th><th>Course</th><th>Tee</th><th>Score</th><th>Putts</th><th>FIR</th><th>GIR</th><th>Rating</th><th>Slope</th><th>Diff</th></tr></thead><tbody>
                {rounds.map((r, i) => (
                  <tr key={i}><td>{fmtDate(r.date)}</td><td style={{ color: C.text, fontWeight: 500, maxWidth: 180, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.courseName || "—"}</td><td>{r.stats?.teeBox || "—"}</td><td style={{ fontWeight: 600, color: C.text }}>{r.score}</td><td>{r.stats?.putts || "—"}</td><td>{pct(r.stats?.fairways, r.stats?.fairwaysPossible)}</td><td>{pct(r.stats?.greens, r.stats?.greensPossible)}</td><td>{r.stats?.courseRating || "—"}</td><td>{r.stats?.slopeRating || "—"}</td><td>{r.differential != null ? fmt(r.differential, 1) : "—"}</td></tr>
                ))}
              </tbody></table>
            </div>
          )}
        </>
      )}
    </section>
  );
}

// ─── Admin ───
function AdminPage({ user }) {
  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [allRounds, setAllRounds] = useState({});
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedUser, setSelectedUser] = useState(null);
  const [selectedRounds, setSelectedRounds] = useState([]);
  const [loadingUser, setLoadingUser] = useState(false);
  const [tab, setTab] = useState("overview");
  const [ocrStats, setOcrStats] = useState({ total: null, last: null });

  useEffect(() => {
    (async () => {
      setLoading(true);
      try {
        const u = await listAllUsers(user.idToken);
        setUsers(u);
        const rm = {};
        await Promise.all(u.map(async (usr) => { try { rm[usr.uid] = await getUserRounds(usr.uid, user.idToken); } catch { rm[usr.uid] = []; } }));
        setAllRounds(rm);
        const ocrBase = {
          from: [{ collectionId: "courses" }],
          where: { fieldFilter: { field: { fieldPath: "source" }, op: "EQUAL", value: { stringValue: "USER_OCR" } } },
        };
        const [totalOcr, lastOcrArr] = await Promise.all([
          firestoreAggregateCount(ocrBase, user.idToken, "ocrCount"),
          firestoreQuery({ ...ocrBase, orderBy: [{ field: { fieldPath: "lastVerifiedAt" }, direction: "DESCENDING" }], limit: 1 }, user.idToken),
        ]);
        setOcrStats({ total: totalOcr, last: lastOcrArr?.[0] || null });
      } catch (e) { console.error(e); }
      setLoading(false);
    })();
  }, [user]);

  const loadUserDetail = async (usr) => {
    setSelectedUser(usr); setLoadingUser(true); setTab("user");
    try { const r = allRounds[usr.uid] || await getUserRounds(usr.uid, user.idToken); setSelectedRounds(r.sort((a, b) => new Date(b.date || 0) - new Date(a.date || 0))); } catch { setSelectedRounds([]); }
    setLoadingUser(false);
  };

  const totalRounds = Object.values(allRounds).reduce((s, r) => s + r.length, 0);
  const activeUsers = Object.entries(allRounds).filter(([, r]) => r.length > 0).length;
  const recentRounds = Object.values(allRounds).flat().filter((r) => daysSince(r.date) <= 7).length;
  const ocrLastAt = ocrStats.last?.lastVerifiedAt ? new Date(ocrStats.last.lastVerifiedAt).toISOString() : (ocrStats.last?.updatedAt || null);
  const ocrLastName = ocrStats.last?.name || "—";

  const filtered = searchTerm ? users.filter((u) => {
    const t = searchTerm.toLowerCase();
    return (u.personalInfo?.name || "").toLowerCase().includes(t) || u.uid.toLowerCase().includes(t);
  }) : users;

  if (loading) return <section style={{ padding: "80px 24px", textAlign: "center" }}><div className="loading-bar" style={{ maxWidth: 200, margin: "40px auto" }} /></section>;

  return (
    <section style={{ padding: "40px 24px 80px", maxWidth: 1200, margin: "0 auto" }}>
      <div className="fade-up" style={{ marginBottom: 32 }}>
        <h1 style={{ fontSize: 24, fontWeight: 700, marginBottom: 4, display: "flex", alignItems: "center", gap: 8 }}>
          <Icon name="tool" size={22} color={C.amber} /> Admin Dashboard
        </h1>
        <p style={{ fontSize: 14, color: C.textMuted }}>Monitor users, rounds, and troubleshoot issues</p>
      </div>

      <div style={{ display: "flex", gap: 4, marginBottom: 20, borderBottom: `1px solid ${C.border}` }}>
        {["overview", "users", ...(selectedUser ? ["user"] : [])].map((t) => (
          <button key={t} onClick={() => setTab(t)} style={{ background: "none", border: "none", padding: "10px 16px", cursor: "pointer", fontFamily: "inherit", fontSize: 14, fontWeight: tab === t ? 600 : 400, color: tab === t ? C.text : C.textMuted, borderBottom: tab === t ? `2px solid ${C.brand}` : "2px solid transparent", textTransform: "capitalize" }}>
            {t === "user" && selectedUser ? `User: ${selectedUser.personalInfo?.name || selectedUser.uid.slice(0, 8)}` : t}
          </button>
        ))}
      </div>

      {tab === "overview" && (
        <div className="fade-in">
          <div className="stat-grid" style={{ marginBottom: 24 }}>
            <div className="stat-box"><div className="stat-value" style={{ color: C.brand }}>{users.length}</div><div className="stat-label">Total Users</div></div>
            <div className="stat-box"><div className="stat-value">{activeUsers}</div><div className="stat-label">Active (1+ round)</div></div>
            <div className="stat-box"><div className="stat-value">{totalRounds}</div><div className="stat-label">Total Rounds</div></div>
            <div className="stat-box"><div className="stat-value">{recentRounds}</div><div className="stat-label">Rounds (7 days)</div></div>
            <div className="stat-box"><div className="stat-value">{users.length ? (totalRounds / users.length).toFixed(1) : "0"}</div><div className="stat-label">Avg Rounds/User</div></div>
            <div className="stat-box"><div className="stat-value">{users.length ? Math.round((activeUsers / users.length) * 100) : 0}%</div><div className="stat-label">Activation Rate</div></div>
            <div className="stat-box"><div className="stat-value">{ocrStats.total != null ? ocrStats.total : "—"}</div><div className="stat-label">OCR Uploads</div></div>
            <div className="stat-box"><div className="stat-value" style={{ fontSize: 13 }}>{ocrLastAt ? fmtDate(ocrLastAt) : "—"}</div><div className="stat-label">Last OCR: {ocrLastName}</div></div>
          </div>
          <div className="card" style={{ padding: 0, overflow: "hidden" }}>
            <div style={{ padding: "16px 20px", borderBottom: `1px solid ${C.border}` }}><h3 style={{ fontSize: 15, fontWeight: 600 }}>Most Active Users</h3></div>
            <table className="table"><thead><tr><th>User</th><th>UID</th><th>Rounds</th><th>Best</th><th>Last Round</th><th>Last Login</th><th></th></tr></thead><tbody>
              {users.sort((a, b) => (allRounds[b.uid]?.length || 0) - (allRounds[a.uid]?.length || 0)).slice(0, 15).map((u, i) => {
                const r = allRounds[u.uid] || [];
                const best = r.length ? Math.min(...r.map((x) => x.score).filter(Boolean)) : null;
                const last = r.length ? r.sort((a, b) => new Date(b.date || 0) - new Date(a.date || 0))[0] : null;
                return (<tr key={i}><td style={{ color: C.text, fontWeight: 500 }}>{u.personalInfo?.name || u.name || u.email || "—"}</td><td style={{ fontSize: 12, fontFamily: "monospace" }}>{u.uid.slice(0, 12)}...</td><td style={{ fontWeight: 600, color: C.text }}>{r.length}</td><td>{best || "—"}</td><td>{last ? fmtDate(last.date) : "—"}</td><td>{u.lastLoginAt ? fmtDate(u.lastLoginAt) : "—"}</td><td><button className="btn btn-ghost btn-sm" onClick={() => loadUserDetail(u)}>View</button></td></tr>);
              })}
            </tbody></table>
          </div>
        </div>
      )}

      {tab === "users" && (
        <div className="fade-in">
          <input className="input" placeholder="Search by name or UID..." value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)} style={{ marginBottom: 16 }} />
          <div className="card" style={{ padding: 0, overflow: "auto" }}>
            <table className="table"><thead><tr><th>Name</th><th>UID</th><th>Mode</th><th>Rounds</th><th>Last Login</th><th>Home Course</th><th></th></tr></thead><tbody>
              {filtered.map((u, i) => (
                <tr key={i}><td style={{ color: C.text, fontWeight: 500 }}>{u.personalInfo?.name || u.name || u.email || "—"}</td><td style={{ fontSize: 12, fontFamily: "monospace" }}>{u.uid.slice(0, 16)}...</td><td><span className={`badge ${u.scoringMode === "advanced" ? "badge-green" : "badge-blue"}`}>{u.scoringMode || "basic"}</span></td><td>{(allRounds[u.uid] || []).length}</td><td>{u.lastLoginAt ? fmtDate(u.lastLoginAt) : "—"}</td><td style={{ maxWidth: 160, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{u.coursePreferences?.homeCourseName || "—"}</td><td><button className="btn btn-ghost btn-sm" onClick={() => loadUserDetail(u)}>Inspect</button></td></tr>
              ))}
            </tbody></table>
          </div>
        </div>
      )}

      {tab === "user" && selectedUser && (
        <div className="fade-in">
          {loadingUser ? <div className="loading-bar" style={{ maxWidth: 200, margin: "40px auto" }} /> : (
            <>
              <div className="card" style={{ marginBottom: 20 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 16 }}>
                  <div>
                    <h3 style={{ fontSize: 18, fontWeight: 700, marginBottom: 4 }}>{selectedUser.personalInfo?.name || selectedUser.name || selectedUser.email || "Unnamed User"}</h3>
                    <p style={{ fontSize: 13, color: C.textMuted, fontFamily: "monospace" }}>UID: {selectedUser.uid}</p>
                  </div>
                  <div style={{ display: "flex", gap: 8 }}>
                    <span className={`badge ${selectedUser.scoringMode === "advanced" ? "badge-green" : "badge-blue"}`}>{selectedUser.scoringMode || "basic"} mode</span>
                    <span className="badge badge-amber">{selectedRounds.length} rounds</span>
                  </div>
                </div>
                <div style={{ marginTop: 16, display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 12 }}>
                  {[
                    ["Home Course", selectedUser.coursePreferences?.homeCourseName],
                    ["Handicap", selectedUser.coursePreferences?.typicalHandicap],
                    ["Favorite Tee", selectedUser.coursePreferences?.favoriteTee],
                    ["Last Login", selectedUser.lastLoginAt ? fmtDate(selectedUser.lastLoginAt) : "—"],
                    ["Device", selectedUser.lastLoginDevice?.platform || selectedUser.lastLoginDevice?.userAgent || "—"],
                    ["Last Error", selectedUser.lastError?.message || "—"],
                  ].map(([l, v], i) => (
                    <div key={i}><span style={{ fontSize: 11, color: C.textDim, textTransform: "uppercase" }}>{l}</span><p style={{ fontSize: 14, color: C.textMuted }}>{v || "—"}</p></div>
                  ))}
                </div>
                {selectedUser.statPreferences && (
                  <div style={{ marginTop: 16 }}>
                    <span style={{ fontSize: 11, color: C.textDim, textTransform: "uppercase" }}>Stat Preferences</span>
                    <div style={{ marginTop: 6, display: "flex", gap: 6, flexWrap: "wrap" }}>
                      {Object.entries(selectedUser.statPreferences).map(([k, v]) => <span key={k} className={`badge ${v ? "badge-green" : "badge-red"}`} style={{ fontSize: 11 }}>{k}</span>)}
                    </div>
                  </div>
                )}
                {selectedUser.goals && Object.values(selectedUser.goals).some((v) => v != null) && (
                  <div style={{ marginTop: 16 }}>
                    <span style={{ fontSize: 11, color: C.textDim, textTransform: "uppercase" }}>Goals</span>
                    <div style={{ marginTop: 6, display: "flex", gap: 12, flexWrap: "wrap" }}>
                      {Object.entries(selectedUser.goals).filter(([, v]) => v != null).map(([k, v]) => (
                        <div key={k} style={{ fontSize: 13, color: C.textMuted }}><span style={{ color: C.text, fontWeight: 600 }}>{v}</span> {k.replace(/([A-Z])/g, " $1").toLowerCase()}</div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
              {selectedRounds.length > 0 && (
                <div className="stat-grid" style={{ marginBottom: 20 }}>
                  {(() => {
                    const sc = selectedRounds.filter((r) => r.score > 0);
                    return [
                      { v: sc.length ? (sc.reduce((s, r) => s + r.score, 0) / sc.length).toFixed(1) : "—", l: "Avg Score" },
                      { v: sc.length ? Math.min(...sc.map((r) => r.score)) : "—", l: "Best" },
                      { v: sc.length ? Math.max(...sc.map((r) => r.score)) : "—", l: "Worst" },
                      { v: sc.filter((r) => r.stats?.putts).length ? (sc.filter((r) => r.stats?.putts).reduce((s, r) => s + r.stats.putts, 0) / sc.filter((r) => r.stats?.putts).length).toFixed(1) : "—", l: "Avg Putts" },
                    ].map((s, i) => <div key={i} className="stat-box"><div className="stat-value">{s.v}</div><div className="stat-label">{s.l}</div></div>);
                  })()}
                </div>
              )}
              <div className="card" style={{ padding: 0, overflow: "auto" }}>
                <div style={{ padding: "16px 20px", borderBottom: `1px solid ${C.border}` }}><h3 style={{ fontSize: 15, fontWeight: 600 }}>Round History</h3></div>
                <table className="table"><thead><tr><th>Date</th><th>Course</th><th>Score</th><th>Putts</th><th>FIR</th><th>GIR</th><th>Tee</th><th>Rating/Slope</th><th>Diff</th><th>Holes</th><th>Status</th></tr></thead><tbody>
                  {selectedRounds.map((r, i) => (
                    <tr key={i}><td>{fmtDate(r.date)}</td><td style={{ color: C.text, fontWeight: 500, maxWidth: 160, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.courseName || "—"}</td><td style={{ fontWeight: 700, color: C.text }}>{r.score}</td><td>{r.stats?.putts || "—"}</td><td>{pct(r.stats?.fairways, r.stats?.fairwaysPossible)}</td><td>{pct(r.stats?.greens, r.stats?.greensPossible)}</td><td>{r.stats?.teeBox || "—"}</td><td>{r.stats?.courseRating || "—"}/{r.stats?.slopeRating || "—"}</td><td>{r.differential != null ? fmt(r.differential, 1) : "—"}</td><td>{r.holeCount || "18"}</td><td>
                      {r.isAcceptableForHandicap === false ? <span className="badge badge-red">Ineligible</span> : r.isIncomplete ? <span className="badge badge-amber">Incomplete</span> : r.isNineHoleRound ? <span className="badge badge-blue">9 holes</span> : <span className="badge badge-green">OK</span>}
                    </td></tr>
                  ))}
                </tbody></table>
              </div>
            </>
          )}
        </div>
      )}
    </section>
  );
}

// ─── Footer ───
function Footer({ nav }) {
  return (
    <footer style={{ borderTop: `1px solid ${C.border}`, padding: "40px 24px", marginTop: 40 }}>
      <div style={{ maxWidth: 1100, margin: "0 auto", display: "flex", justifyContent: "space-between", flexWrap: "wrap", gap: 24 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
            <div style={{ width: 24, height: 24, borderRadius: 6, background: C.brand, display: "flex", alignItems: "center", justifyContent: "center" }}><Icon name="flag" size={12} color="#fff" /></div>
            <span style={{ fontWeight: 700, fontSize: 15 }}>GolfSum</span>
          </div>
          <p style={{ fontSize: 13, color: C.textDim }}>Track · Analyze · Improve</p>
        </div>
        <div style={{ display: "flex", gap: 32 }}>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <span style={{ fontSize: 11, fontWeight: 600, color: C.textDim, textTransform: "uppercase", letterSpacing: "0.06em" }}>Product</span>
            <a href="#" onClick={(e) => { e.preventDefault(); nav("features"); }} style={{ fontSize: 14, color: C.textMuted }}>Features</a>
            <a href="#" onClick={(e) => { e.preventDefault(); nav("pricing"); }} style={{ fontSize: 14, color: C.textMuted }}>Pricing</a>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <span style={{ fontSize: 11, fontWeight: 600, color: C.textDim, textTransform: "uppercase", letterSpacing: "0.06em" }}>Legal</span>
            <a href="#" onClick={(e) => { e.preventDefault(); nav("privacy"); }} style={{ fontSize: 14, color: C.textMuted }}>Privacy Policy</a>
            <a href="#" onClick={(e) => { e.preventDefault(); nav("terms"); }} style={{ fontSize: 14, color: C.textMuted }}>Terms of Service</a>
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <span style={{ fontSize: 11, fontWeight: 600, color: C.textDim, textTransform: "uppercase", letterSpacing: "0.06em" }}>Support</span>
            <a href="mailto:support@golfsum.com" style={{ fontSize: 14, color: C.textMuted }}>support@golfsum.com</a>
          </div>
        </div>
      </div>
      <div style={{ maxWidth: 1100, margin: "24px auto 0", borderTop: `1px solid ${C.border}`, paddingTop: 20 }}>
        <p style={{ fontSize: 12, color: C.textDim }}>© 2026 GolfSum. All rights reserved.</p>
      </div>
    </footer>
  );
}

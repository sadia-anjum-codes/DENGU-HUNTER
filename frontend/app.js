
'use strict';

/* =========================================================
   DENGU HUNTER — LIVE FRONTEND
   Existing FastAPI + Existing Trained YOLO
========================================================= */

const S = {
    page: 'dashboard',
    caseId: null,
    cases: [],
    dashboard: null,
    health: null,
    token: '',
    base: 'http://127.0.0.1:8003',
    busy: false,
    search: '',
    priority: 'ALL',
    stage: 'ALL',
    file: null,
    afterFile: null,
    preview: null,
    tab: 'overview',
    lastCase: null,
    evidenceUrls: [],
    map: null
};

const $ = s => document.querySelector(s);

const esc = v => String(v ?? '—').replace(/[&<>"']/g, c => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
}[c]));

const nav = [
    ['dashboard', '▦', 'Dashboard'],
    ['report', '⊕', 'New Report / AI'],
    ['cases', '▤', 'Case Management'],
    ['map', '⌖', 'GPS Surveillance Map'],
    ['drone', '✈', 'Drone Mission'],
    ['tasks', '▣', 'Worker Tasks'],
    ['review', '☑', 'Human Review'],
    ['after', '⇄', 'After & Finalize'],
    ['analytics', '◷', 'Analytics'],
    ['alerts', '♧', 'Alerts'],
    ['settings', '⚙', 'Connection & Settings']
];

/* =========================================================
   COMMON FUNCTIONS
========================================================= */

function toast(message, error = false) {
    const n = $('#notice');

    if (!n) return;

    n.hidden = false;
    n.textContent = message;
    n.style.borderColor = error ? '#a63c5b' : '#5b3b87';

    clearTimeout(toast.timer);

    toast.timer = setTimeout(() => {
        n.hidden = true;
    }, 7000);
}

function tag(v) {
    const x = String(v ?? 'UNKNOWN');

    let cl = /HIGH|REJECT|UNRESOLVED/.test(x)
        ? 'high'
        : /LOW|APPROV|RESOLVED|CLOSED/.test(x)
        ? 'low'
        : /MEDIUM|PENDING|REINSPECTION/.test(x)
        ? 'medium'
        : 'blue';

    return `<span class="tag ${cl}">
        ${esc(x.replaceAll('_', ' '))}
    </span>`;
}

function btn(t, act, cls = '') {
    return `<button class="btn ${cls}" data-action="${act}">
        ${t}
    </button>`;
}

function box(title, body, extra = '') {
    return `
        <section class="panel">
            <div class="panel-head">
                <h2>${title}</h2>
                ${extra}
            </div>
            ${body}
        </section>
    `;
}

function detail(c) {
    return `
        <div class="detailgrid">
            ${[
                ['Case ID', c.case_id],
                ['Workflow Stage', c.stage],
                ['Review', c.review_status],
                ['Priority', c.priority?.level],
                ['Priority Score', c.priority?.score ?? 0],
                ['Worker', c.assigned_worker || 'Not assigned'],
                ['Task', c.task_id || 'Not created'],
                ['Evidence', c.evidence_mode],
                [
                    'GPS',
                    c.gps_provided &&
                    c.latitude != null &&
                    c.longitude != null
                        ? `${c.latitude}, ${c.longitude}`
                        : 'Not provided'
                ],
                ['Field Verified', c.real_field_verified ? 'YES' : 'NO']
            ].map(([k, v]) => `
                <div>
                    <small>${esc(k)}</small>
                    <b>${esc(v)}</b>
                </div>
            `).join('')}
        </div>
    `;
}

function detections(c) {
    const ds = c.f2_detections || [];

    return ds.length
        ? ds.map(d => `
            <div class="listrow">
                <span>
                    <b>${esc(d.class_name)}</b>
                    <br>
                    <span class="muted smalltext">
                        YOLO object detection
                    </span>
                </span>

                <span class="accent">
                    <b>${(100 * (d.confidence || 0)).toFixed(1)}%</b>
                </span>
            </div>
        `).join('')
        : `
            <div class="empty">
                No visible objects detected by the model.
                Human inspection may still be necessary.
            </div>
        `;
}

function selectCase(id) {
    S.caseId = id;
    S.page = 'detail';
    S.tab = 'overview';
    render();
}

async function api(path, opt = {}) {
    if (!S.token) {
        throw Error('Connect your API token in Settings first.');
    }

    const h = {
        Authorization: `Bearer ${S.token}`,
        ...(opt.body instanceof FormData
            ? {}
            : { 'Content-Type': 'application/json' })
    };

    let r;

    try {
        r = await fetch(S.base + '/api/v1' + path, {
            ...opt,
            headers: {
                ...h,
                ...opt.headers
            }
        });
    } catch (e) {
        throw Error(
            'Cannot reach backend. Start Uvicorn and check port 8003.'
        );
    }

    const raw = await r.text();

    let data;

    try {
        data = JSON.parse(raw);
    } catch {
        data = { detail: raw };
    }

    if (!r.ok) {
        const msg = typeof data.detail === 'string'
            ? data.detail
            : JSON.stringify(data.detail || data);

        throw Error(`HTTP ${r.status}: ${msg}`);
    }

    return data;
}

async function sync() {
    if (!S.token) {
        render();
        return;
    }

    try {
        const [cases, db, health] = await Promise.all([
            api('/cases'),
            api('/dashboard'),
            api('/health')
        ]);

        S.cases = Array.isArray(cases) ? cases : [];
        S.dashboard = db;
        S.health = health;

        const connection = $('#connection');

        if (connection) {
            connection.textContent = 'Online · YOLO';
        }

        render();

    } catch (e) {
        const connection = $('#connection');

        if (connection) {
            connection.textContent = 'Disconnected';
        }

        toast(e.message, true);
        render();
    }
}

function go(page) {
    S.page = page;
    S.search = '';
    render();

    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

function stat(label, value, icon, sub) {
    return `
        <div class="stat">
            <span class="stat-icon">${icon}</span>
            <div class="label">${label}</div>
            <div class="value">${value}</div>
            <div class="mini">${sub}</div>
        </div>
    `;
}

function count(fn) {
    return S.cases.filter(fn).length;
}

/* =========================================================
   DASHBOARD
========================================================= */

function chart() {
    const vals = [0, 0, 0, 0, 0, 0, 0];

    S.cases.forEach((c, i) => {
        vals[i % 7]++;
    });

    const max = Math.max(...vals, 1);

    const pts = vals.map((v, i) =>
        `${30 + i * 68},${166 - v / max * 120}`
    ).join(' ');

    return `
        <svg class="chart" viewBox="0 0 470 200" role="img">
            ${[0, 1, 2, 3].map(i => `
                <line
                    x1="25"
                    y1="${32 + i * 44}"
                    x2="460"
                    y2="${32 + i * 44}"
                />
            `).join('')}

            <polyline
                points="${pts}"
                fill="none"
                stroke="#8b5cf6"
                stroke-width="3"
                stroke-linecap="round"
                stroke-linejoin="round"
            />

            ${pts.split(' ').map(p => {
                const [x, y] = p.split(',');

                return `
                    <circle
                        cx="${x}"
                        cy="${y}"
                        r="4"
                        fill="#a78bfa"
                    />
                `;
            }).join('')}

            ${vals.map((v, i) => `
                <text
                    x="${30 + i * 68}"
                    y="191"
                    text-anchor="middle"
                >${i + 1}</text>
            `).join('')}
        </svg>

        <p class="smalltext">
            Illustrative distribution of loaded cases,
            not a historical trend.
        </p>
    `;
}

function caseTable(rows = S.cases) {
    return rows.length
        ? `
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>CASE ID</th>
                            <th>AI OBJECT</th>
                            <th>PRIORITY</th>
                            <th>STAGE</th>
                            <th>WORKER</th>
                            <th>ACTION</th>
                        </tr>
                    </thead>

                    <tbody>
                        ${rows.map(c => `
                            <tr>
                                <td>
                                    <button
                                        class="link"
                                        data-case="${esc(c.case_id)}"
                                    >
                                        ${esc(c.case_id)}
                                    </button>
                                </td>

                                <td>
                                    ${esc(
                                        c.f2_detections?.[0]?.class_name
                                        || 'Unconfirmed'
                                    )}
                                </td>

                                <td>${tag(c.priority?.level)}</td>
                                <td>${tag(c.stage)}</td>

                                <td>
                                    ${esc(c.assigned_worker || '—')}
                                </td>

                                <td>
                                    <button
                                        class="btn small"
                                        data-case="${esc(c.case_id)}"
                                    >
                                        View →
                                    </button>
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `
        : `
            <div class="empty">
                No matching cases. Connect your backend or submit a new report.
            </div>
        `;
}

function dashboard() {
    const cs = S.cases;
    const latest = cs[0];

    return `
        <div class="hero">
            <div class="kicker">
                ◈ INTELLIGENT FIELD RESPONSE
            </div>

            <h2>Welcome to DENGU HUNTER</h2>

            <p>
                Computer vision, human review, and accountable
                field action — one command center.
            </p>

            <div class="actions">
                ${btn('＋ Create Report', 'newreport', 'primary')}
                ${btn('View All Cases', 'allcases')}
                ${btn('⌖ Open GPS Map', 'openmap')}
            </div>
        </div>

        <div class="grid stats">
            ${stat('Total Reports', cs.length, '▤', 'Loaded from live API')}

            ${stat(
                'High Priority',
                count(c => c.priority?.level === 'HIGH'),
                '⚠',
                'Inspection priority only'
            )}

            ${stat(
                'Tasks In Progress',
                count(c => c.task_status === 'IN_PROGRESS'),
                '▣',
                'Assigned field operations'
            )}

            ${stat(
                'Evidence Reviewed',
                count(c => c.stage === 'CLOSED_EVIDENCE_REVIEWED'),
                '◇',
                'Not field-verified cleanup'
            )}
        </div>

        <div class="grid two">
            ${box(
                '◉ AI Detection Overview',
                latest
                    ? `
                        <div class="detection">
                            <div class="preview">
                                <span class="symbol">◎</span>
                                <span class="boxlabel">YOLO F2</span>
                            </div>

                            <div>
                                <div class="muted smalltext">
                                    Most recent case
                                </div>

                                <h2>
                                    ${esc(
                                        latest.f2_detections?.[0]?.class_name
                                        || 'No detection'
                                    )}
                                </h2>

                                <div class="accent">
                                    ${latest.f2_detections?.length
                                        ? (
                                            (latest.f2_detections[0].confidence || 0)
                                            * 100
                                        ).toFixed(1) + '% confidence'
                                        : 'Awaiting inspection'
                                    }
                                </div>

                                <div class="progress">
                                    <div style="width:${
                                        Math.min(
                                            100,
                                            (latest.f2_detections?.[0]?.confidence || 0) * 100
                                        )
                                    }%"></div>
                                </div>

                                <p class="smalltext">
                                    Detection identifies possible
                                    water-holding objects; dengue is not confirmed.
                                </p>

                                <div class="actions">
                                    <button
                                        class="btn small"
                                        data-case="${esc(latest.case_id)}"
                                    >
                                        Inspect Case →
                                    </button>
                                </div>
                            </div>
                        </div>
                    `
                    : `
                        <div class="empty">
                            Upload a ground photo to start YOLO detection.
                        </div>
                    `
            )}

            ${box('◷ Case Distribution', chart())}
        </div>

        <div class="grid two" style="margin-top:17px">
            ${box(
                'Recent Cases',
                caseTable(cs.slice(0, 5)),
                btn('View all →', 'allcases', 'small')
            )}

            ${box(
                'Quick Actions',
                `
                    <div class="stack">
                        ${btn('＋ New Report · Run AI', 'newreport', 'primary full')}
                        ${btn('⌖ GPS Surveillance Map', 'openmap', 'full')}
                        ${btn('☑ Review Pending Cases', 'pendingreview', 'full')}
                        ${btn('▣ Manage Worker Tasks', 'workertasks', 'full')}
                        ${btn('⇄ After Evidence & Final Review', 'afterpage', 'full')}
                    </div>
                `
            )}
        </div>
    `;
}

/* =========================================================
   NEW REPORT AND LIVE YOLO INFERENCE
========================================================= */

function report() {
    return `
        <div class="hero">
            <div class="kicker">F2 · LIVE YOLO MODEL</div>

            <h2>New Breeding-Site Inspection Report</h2>

            <p>
                Upload a photo of a potential water-holding object.
                The AI result is a screening aid, not mosquito-species identification.
            </p>
        </div>

        <div class="split">

            ${box(
                '01 / Upload Site Evidence',
                `
                    <form id="reportForm" class="stack">

                        <div class="field">
                            <label for="photo">
                                Ground photograph (JPG / PNG, max 12 MB)
                            </label>

                            <input
                                id="photo"
                                name="image"
                                type="file"
                                accept="image/png,image/jpeg"
                                required
                            >
                        </div>

                        <div id="uploadPreview" class="preview wide">
                            <span class="symbol">⇧</span>
                        </div>

                        <div class="formgrid">

                            <div class="field">
                                <label>Evidence classification</label>
                                <select name="evidence_mode">
                                    <option value="DEMO_CASE_STAGED">
                                        Staged demonstration
                                    </option>
                                    <option value="FIELD_REPORT_UNVERIFIED">
                                        Unverified field report
                                    </option>
                                </select>
                            </div>

                            <div class="field">
                                <label>Image source</label>
                                <select name="source">
                                    <option>Ground photo</option>
                                    <option>Aerial photo (user-provided)</option>
                                    <option>Drone simulation frame</option>
                                </select>
                            </div>

                            <div class="field">
                                <label>Rain context</label>
                                <select name="rain_context">
                                    <option>Unknown</option>
                                    <option>Dry</option>
                                    <option>Recent rain</option>
                                    <option>Heavy rain</option>
                                </select>
                            </div>

                            <div class="field">
                                <label>Previous reports</label>
                                <input
                                    name="previous_reports"
                                    type="number"
                                    min="0"
                                    value="0"
                                >
                            </div>

                            <div class="field">
                                <label>Latitude (optional)</label>
                                <input
                                    name="latitude"
                                    type="number"
                                    step="any"
                                    min="-90"
                                    max="90"
                                    placeholder="Only if verified"
                                >
                            </div>

                            <div class="field">
                                <label>Longitude (optional)</label>
                                <input
                                    name="longitude"
                                    type="number"
                                    step="any"
                                    min="-180"
                                    max="180"
                                    placeholder="Only if verified"
                                >
                            </div>
                        </div>

                        <div class="field">
                            <label>Inspector notes</label>
                            <textarea
                                name="notes"
                                placeholder="Describe the visible site without claiming dengue confirmation"
                            ></textarea>
                        </div>

                        <button class="btn primary full" type="submit">
                            ◉ Analyze with Live YOLO AI
                        </button>

                    </form>
                `
            )}

            ${box(
                '02 / Detection Guidance',
                `
                    <div class="stack">

                        <div class="listrow">
                            <span>◈ <b>Model</b></span>
                            <span class="accent">Existing trained F2 YOLO</span>
                        </div>

                        <div class="listrow">
                            <span>◇ <b>Detectable objects</b></span>
                            <span class="muted">
                                Tire, vase, bottle, coconut, drain inlet
                            </span>
                        </div>

                        <div class="listrow">
                            <span>▤ <b>Priority</b></span>
                            <span class="muted">Explainable heuristic</span>
                        </div>

                        <div class="listrow">
                            <span>☑ <b>Next</b></span>
                            <span class="muted">Human review</span>
                        </div>

                        <p>
                            F3 larvae model is not configured.
                            No claim of Aedes species or dengue presence is made.
                        </p>

                        ${S.lastCase
                            ? `
                                <div class="notice">
                                    Created ${esc(S.lastCase.case_id)}
                                    · ${esc(S.lastCase.priority?.level)} priority.
                                </div>

                                ${btn(
                                    'Open Created Case',
                                    'lastcase',
                                    'primary'
                                )}
                            `
                            : ''
                        }
                    </div>
                `
            )}

        </div>
    `;
}

/* =========================================================
   CASE MANAGEMENT
========================================================= */

function cases() {
    const cs = S.cases.filter(c =>
        (
            !S.search ||
            JSON.stringify([
                c.case_id,
                c.stage,
                c.assigned_worker,
                c.f2_detections?.[0]?.class_name
            ]).toLowerCase().includes(S.search.toLowerCase())
        ) &&
        (
            S.priority === 'ALL' ||
            c.priority?.level === S.priority
        ) &&
        (
            S.stage === 'ALL' ||
            c.stage === S.stage
        )
    );

    return box(
        'Case Registry',

        `
            <div class="filters">

                <input
                    id="caseSearch"
                    placeholder="Search case ID, object, worker, or stage..."
                    value="${esc(S.search)}"
                >

                <select id="priorityFilter">
                    <option value="ALL">All priorities</option>

                    ${['HIGH', 'MEDIUM', 'LOW', 'UNCONFIRMED'].map(x => `
                        <option
                            value="${x}"
                            ${S.priority === x ? 'selected' : ''}
                        >${x}</option>
                    `).join('')}
                </select>

                <select id="stageFilter">
                    <option value="ALL">All stages</option>

                    ${[...new Set(S.cases.map(c => c.stage))].map(x => `
                        <option
                            value="${esc(x)}"
                            ${S.stage === x ? 'selected' : ''}
                        >${esc(x)}</option>
                    `).join('')}
                </select>

            </div>

            <div id="caseResults">
                ${caseTable(cs)}
            </div>
        `,

        `<span class="pill">${cs.length} / ${S.cases.length} CASES</span>`
    );
}

function chooseRow(c) {
    return `
        <div class="listrow">

            <div>
                <button
                    class="link"
                    data-case="${esc(c.case_id)}"
                >
                    ${esc(c.case_id)}
                </button>

                <div class="muted smalltext">
                    ${esc(c.f2_detections?.[0]?.class_name || 'Unconfirmed')}
                    · ${esc(c.assigned_worker || 'Unassigned')}
                </div>
            </div>

            <div>
                ${tag(c.stage)}
                ${btn('Open', 'open:' + c.case_id, 'small')}
            </div>

        </div>
    `;
}

/* =========================================================
   GPS SURVEILLANCE MAP
========================================================= */

function validGPS(c) {
    const lat = Number(c.latitude);
    const lon = Number(c.longitude);

    return (
        c.gps_provided === true &&
        c.latitude != null &&
        c.longitude != null &&
        Number.isFinite(lat) &&
        Number.isFinite(lon) &&
        lat >= -90 &&
        lat <= 90 &&
        lon >= -180 &&
        lon <= 180 &&
        !(lat === 0 && lon === 0)
    );
}

function mapPage() {
    const located = S.cases.filter(validGPS);
    const missing = S.cases.length - located.length;

    return `
        <div class="hero">
            <div class="kicker">◈ LOCATION-AWARE SURVEILLANCE</div>

            <h2>GPS Surveillance Map</h2>

            <p>
                Explore reported breeding-site inspection locations
                using their submitted GPS coordinates.
            </p>

            <div class="actions">
                ${btn('↻ Refresh Map', 'refreshmap')}
                ${btn('View All Cases', 'allcases')}
            </div>
        </div>

        <div class="grid stats">

            ${stat(
                'Total Cases',
                S.cases.length,
                '▤',
                'Live backend records'
            )}

            ${stat(
                'GPS Locations',
                located.length,
                '⌖',
                'Coordinates available'
            )}

            ${stat(
                'Without GPS',
                missing,
                '◇',
                'Location not provided'
            )}

            ${stat(
                'High Priority',
                located.filter(c => c.priority?.level === 'HIGH').length,
                '⚠',
                'Located high-priority cases'
            )}

        </div>

        <div class="panel">

            <div class="panel-head">
                <h2>Interactive Inspection Map</h2>

                <span class="pill">
                    ${located.length} GPS MARKERS
                </span>
            </div>

            <div
                id="gpsMap"
                style="
                    width:100%;
                    height:600px;
                    min-height:400px;
                    border-radius:12px;
                    overflow:hidden;
                    background:#18151f;
                    border:1px solid #37303f;
                "
            >
                Loading map...
            </div>

            <p class="smalltext" style="margin-top:12px">
                Map uses actual submitted GPS coordinates.
                Cases without GPS are not assigned artificial locations.
            </p>

        </div>

        ${box(
            'GPS Location Registry',
            located.length
                ? `
                    <div class="table-wrap">
                        <table>
                            <thead>
                                <tr>
                                    <th>CASE ID</th>
                                    <th>LATITUDE</th>
                                    <th>LONGITUDE</th>
                                    <th>PRIORITY</th>
                                    <th>STAGE</th>
                                    <th>ACTION</th>
                                </tr>
                            </thead>

                            <tbody>
                                ${located.map(c => `
                                    <tr>
                                        <td>${esc(c.case_id)}</td>
                                        <td>${Number(c.latitude).toFixed(6)}</td>
                                        <td>${Number(c.longitude).toFixed(6)}</td>
                                        <td>${tag(c.priority?.level)}</td>
                                        <td>${tag(c.stage)}</td>
                                        <td>
                                            <button
                                                class="btn small"
                                                data-case="${esc(c.case_id)}"
                                            >
                                                View →
                                            </button>
                                        </td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                `
                : `
                    <div class="empty">
                        No cases have verified GPS coordinates yet.
                        Add latitude and longitude when submitting a report.
                    </div>
                `
        )}loadGPSMap
    `;
}

/* Load Leaflet only when the map page is opened */

function loadLeaflet() {
    return new Promise((resolve, reject) => {
        if (window.L) {
            resolve();
            return;
        }

        if (!document.querySelector('#leaflet-css')) {
            const css = document.createElement('link');

            css.id = 'leaflet-css';
            css.rel = 'stylesheet';
            css.href =
                'https://unpkg.com/leaflet@1.9.4/dist/leaflet.css';

            document.head.appendChild(css);
        }

        const existing = document.querySelector('#leaflet-js');

        if (existing) {
            existing.addEventListener('load', resolve, { once: true });
            existing.addEventListener('error', reject, { once: true });

            if (window.L) resolve();

            return;
        }

        const script = document.createElement('script');

        script.id = 'leaflet-js';
        script.src =
            'https://unpkg.com/leaflet@1.9.4/dist/leaflet.js';

        script.onload = resolve;
        script.onerror = () => reject(
            Error('Cannot load Leaflet. Check your internet connection.')
        );

        document.head.appendChild(script);
    });
}


async function loadGPSMap() {
    const target = $('#gpsMap');

    if (!target || S.page !== 'map') return;

    const located = S.cases.filter(validGPS);

    try {
        await loadLeaflet();

        if (S.page !== 'map' || !target.isConnected) return;

        // Remove the previous map before creating a new one.
        if (S.map) {
            S.map.remove();
            S.map = null;
        }

        target.replaceChildren();

        const center = located.length
            ? [
                Number(located[0].latitude),
                Number(located[0].longitude)
            ]
            : [23.685, 90.3563];

        const map = L.map(target).setView(
            center,
            located.length ? 12 : 7
        );

        S.map = map;

        L.tileLayer(
            'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
            {
                maxZoom: 19,
                attribution: '&copy; OpenStreetMap contributors'
            }
        ).addTo(map);

        const colors = {
            HIGH: '#ef4444',
            MEDIUM: '#f59e0b',
            LOW: '#22c55e',
            UNCONFIRMED: '#94a3b8'
        };

        const markers = [];

        located.forEach(c => {
            const lat = Number(c.latitude);
            const lon = Number(c.longitude);

            const color =
                colors[c.priority?.level] || '#8b5cf6';

            const marker = L.circleMarker(
                [lat, lon],
                {
                    radius: 9,
                    color: color,
                    fillColor: color,
                    fillOpacity: 0.85,
                    weight: 2
                }
            ).addTo(map);

            marker.bindPopup(`
                <div style="min-width:180px">
                    <b>${esc(c.case_id)}</b>
                    <hr>
                    <div>Object:
                        ${esc(c.f2_detections?.[0]?.class_name || 'Unknown')}
                    </div>
                    <div>Priority:
                        ${esc(c.priority?.level)}
                    </div>
                    <div>Score:
                        ${esc(c.priority?.score)}
                    </div>
                    <div>Stage:
                        ${esc(c.stage)}
                    </div>
                    <div>Worker:
                        ${esc(c.assigned_worker || 'Unassigned')}
                    </div>
                    <div style="margin-top:6px">
                        ${lat.toFixed(6)}, ${lon.toFixed(6)}
                    </div>
                </div>
            `);

            markers.push(marker);
        });

        if (markers.length > 1) {
            const group = L.featureGroup(markers);

            map.fitBounds(group.getBounds(), {
                padding: [35, 35],
                maxZoom: 14
            });
        }

        // IMPORTANT: Do not modify target.innerHTML
        // after Leaflet has initialized the map.
        if (!located.length) {
            const notice = document.createElement('div');

            notice.textContent =
                'No GPS coordinates available yet.';

            Object.assign(notice.style, {
                position: 'absolute',
                top: '12px',
                left: '45px',
                zIndex: '500',
                padding: '12px',
                background: '#17131f',
                color: 'white',
                border: '1px solid #5b3b87',
                borderRadius: '8px',
                pointerEvents: 'none'
            });

            target.appendChild(notice);
        }

        // Update size only while this map is active.
        setTimeout(() => {
            if (
                S.map === map &&
                target.isConnected &&
                S.page === 'map'
            ) {
                map.invalidateSize();
            }
        }, 200);

    } catch (e) {
        if (S.map) {
            try {
                S.map.remove();
            } catch (_) {}

            S.map = null;
        }

        if (target.isConnected) {
            target.replaceChildren();

            const errorBox = document.createElement('div');
            errorBox.className = 'empty';
            errorBox.style.padding = '30px';
            errorBox.textContent = e.message;

            target.appendChild(errorBox);
        }

        toast(e.message, true);
    }
}


/* =========================================================
   WORKER TASKS
========================================================= */

function tasks() {
    const rows = S.cases.filter(c =>
        c.task_id ||
        c.stage === 'TASK_READY' ||
        c.stage === 'WORK_IN_PROGRESS'
    );

    return `
        <div class="grid stats">

            ${stat(
                'Task Ready',
                count(c => c.stage === 'TASK_READY'),
                '▣',
                'Ready for assignment'
            )}

            ${stat(
                'Assigned',
                count(c => c.stage === 'TASK_ASSIGNED'),
                '◉',
                'Awaiting start'
            )}

            ${stat(
                'In Progress',
                count(c => c.stage === 'WORK_IN_PROGRESS'),
                '↻',
                'Field activity'
            )}

            ${stat(
                'After Photo Submitted',
                count(c => c.stage === 'AWAITING_FINAL_HUMAN_REVIEW'),
                '⇄',
                'Needs final review'
            )}

        </div>

        ${box(
            'Worker Operations',
            rows.length
                ? rows.map(chooseRow).join('')
                : `
                    <div class="empty">
                        No assigned tasks yet.
                        Approve a case to create one.
                    </div>
                `
        )}
    `;
}

/* =========================================================
   HUMAN REVIEW
========================================================= */

function review() {
    const rows = S.cases.filter(
        c => c.stage === 'AWAITING_HUMAN_REVIEW'
    );

    return `
        <div class="hero">
            <h2>Human-in-the-Loop Review</h2>

            <p>
                Every AI detection requires a reviewer
                before a task can be assigned.
            </p>
        </div>

        ${box(
            'Awaiting Initial Review',
            rows.length
                ? rows.map(chooseRow).join('')
                : `
                    <div class="empty">
                        No cases currently awaiting initial review.
                    </div>
                `
        )}

        ${box(
            'Other Cases',
            S.cases
                .filter(c => c.stage !== 'AWAITING_HUMAN_REVIEW')
                .slice(0, 6)
                .map(chooseRow)
                .join('') || '<p>None</p>'
        )}
    `;
}

/* =========================================================
   AFTER EVIDENCE QUEUE
========================================================= */

function after() {
    const rows = S.cases.filter(c =>
        [
            'WORK_IN_PROGRESS',
            'AWAITING_FINAL_HUMAN_REVIEW',
            'REINSPECTION_REQUIRED',
            'CLOSED_EVIDENCE_REVIEWED',
            'UNRESOLVED'
        ].includes(c.stage)
    );

    return `
        <div class="hero">
            <h2>Before → After → Human Final Review</h2>

            <p>
                Worker same-site attestation is a declaration,
                not independent proof.
                Final review never automatically claims field-verified cleanup.
            </p>
        </div>

        ${box(
            'Field Evidence Queue',
            rows.length
                ? rows.map(chooseRow).join('')
                : `
                    <div class="empty">
                        No cases at the after-evidence stage.
                    </div>
                `
        )}
    `;
}

/* =========================================================
   ANALYTICS
========================================================= */

function analytics() {
    const cs = S.cases;

    const levels = [
        'HIGH',
        'MEDIUM',
        'LOW',
        'UNCONFIRMED'
    ];

    const stages = [...new Set(cs.map(c => c.stage))];

    return `
        <div class="grid stats">

            ${stat('Cases', cs.length, '▤', 'Live API data')}

            ${stat(
                'Model Detections',
                cs.reduce(
                    (a, c) => a + (c.f2_detections?.length || 0),
                    0
                ),
                '◉',
                'Detected objects'
            )}

            ${stat(
                'Reviewed',
                count(c =>
                    c.review_status === 'APPROVE' ||
                    c.review_status === 'REJECT'
                ),
                '☑',
                'Human decisions'
            )}

            ${stat(
                'GPS Locations',
                cs.filter(validGPS).length,
                '⌖',
                'Cases with coordinates'
            )}

        </div>

        <div class="grid two">

            ${box(
                'Inspection Priority Distribution',
                levels.map(l => `
                    <div class="listrow">
                        <span>${tag(l)}</span>
                        <b>${count(c => c.priority?.level === l)}</b>
                    </div>
                `).join('')
            )}

            ${box(
                'Workflow Stages',
                stages.map(l => `
                    <div class="listrow">
                        <span class="smalltext">${esc(l)}</span>
                        <b>${count(c => c.stage === l)}</b>
                    </div>
                `).join('') || '<p>No cases loaded.</p>'
            )}

        </div>

        <div class="panel" style="margin-top:17px">
            <h2>Interpretation</h2>

            <p>
                Priority scores are rule-based inspection triage,
                not dengue forecasts.
                This view summarizes available local cases;
                it is not an epidemiological trend.
            </p>
        </div>
    `;
}

/* =========================================================
   ALERTS
========================================================= */

function alerts() {
    const high = S.cases.filter(c =>
        c.priority?.level === 'HIGH' &&
        !/CLOSED|RESOLVED/.test(c.stage)
    );

    const pending = S.cases.filter(
        c => c.stage === 'AWAITING_HUMAN_REVIEW'
    );

    const afters = S.cases.filter(
        c => c.stage === 'AWAITING_FINAL_HUMAN_REVIEW'
    );

    return `
        ${box(
            'Actionable Queue',
            `
                <div class="listrow">
                    <b class="dangertext">
                        ⚠ High-priority inspections
                    </b>
                    <b>${high.length}</b>
                </div>

                ${high.map(chooseRow).join('')}

                <div class="listrow">
                    <b class="accent">
                        ☑ Awaiting human review
                    </b>
                    <b>${pending.length}</b>
                </div>

                ${pending.map(chooseRow).join('')}

                <div class="listrow">
                    <b class="warn">
                        ⇄ Awaiting final evidence review
                    </b>
                    <b>${afters.length}</b>
                </div>

                ${afters.map(chooseRow).join('')}
            `
        )}

        <p style="margin-top:12px">
            Alerts are derived from the loaded case list,
            not push notifications or a live external alert service.
        </p>
    `;
}

/* =========================================================
   SETTINGS
========================================================= */

function settings() {
    return `
        <div class="split">

            ${box(
                'Connect to Local FastAPI',
                `
                    <form id="settingsForm" class="stack">

                        <div class="field">
                            <label>Backend API URL</label>

                            <input
                                id="apiBase"
                                type="url"
                                value="${esc(S.base)}"
                                required
                            >
                        </div>

                        <div class="field">
                            <label>Bearer token (from backend .env)</label>

                            <input
                                id="apiToken"
                                type="password"
                                autocomplete="off"
                                placeholder="Paste DENGU_API_TOKEN"
                                value=""
                                required
                            >
                        </div>

                        <p class="smalltext">
                            Token stays only in this browser tab's memory.
                            It is not included in the ZIP, URL, or localStorage.
                            Use this only on your own machine.
                        </p>

                        <button class="btn primary full" type="submit">
                            Connect & Load Live Data
                        </button>

                        ${btn('Disconnect', 'disconnect', 'full')}

                    </form>
                `
            )}

            ${box(
                'API & Model Status',
                `
                    <div class="listrow">
                        <span>Connection</span>
                        <b>${S.health ? '● ONLINE' : '○ NOT CONNECTED'}</b>
                    </div>

                    <div class="listrow">
                        <span>Model</span>
                        <b>${esc(S.health?.model || '—')}</b>
                    </div>

                    <div class="listrow">
                        <span>Backend Status</span>
                        <b>${esc(S.health?.status || '—')}</b>
                    </div>

                    <div class="listrow">
                        <span>Loaded Cases</span>
                        <b>${S.cases.length}</b>
                    </div>

                    <p style="margin-top:14px">
                        Run this frontend on port 5173 to match
                        the backend's existing CORS allowlist.
                        Never publish a shared admin token
                        in a public browser application.
                    </p>
                `
            )}

        </div>
    `;
}

/* =========================================================
   REAL YOLO EVIDENCE IMAGES
========================================================= */

function clearEvidenceUrls() {
    S.evidenceUrls.forEach(url => URL.revokeObjectURL(url));
    S.evidenceUrls = [];
}

async function loadEvidenceImages() {
    if (
        !S.token ||
        !S.caseId ||
        S.page !== 'detail' ||
        S.tab !== 'ai'
    ) {
        return;
    }

    const caseId = S.caseId;

    const types = [
        {
            key: 'before',
            title: 'Original Ground Photo'
        },
        {
            key: 'before_overlay',
            title: 'Actual YOLO Prediction'
        }
    ];

    for (const item of types) {
        const target = $(`#evidence-${item.key}`);

        if (!target) continue;

        target.innerHTML = `
            <div class="empty">Loading ${item.title}...</div>
        `;

        try {
            const response = await fetch(
                `${S.base}/api/v1/cases/${encodeURIComponent(caseId)}/images/${item.key}`,
                {
                    headers: {
                        Authorization: `Bearer ${S.token}`
                    },
                    cache: 'no-store'
                }
            );

            if (!response.ok) {
                throw Error(
                    `Image API returned HTTP ${response.status}`
                );
            }

            const blob = await response.blob();

            if (!blob.type.startsWith('image/')) {
                throw Error('Backend did not return an image.');
            }

            const imageURL = URL.createObjectURL(blob);

            S.evidenceUrls.push(imageURL);

            if (
                S.page !== 'detail' ||
                S.tab !== 'ai' ||
                S.caseId !== caseId
            ) {
                URL.revokeObjectURL(imageURL);
                continue;
            }

            const currentTarget = $(`#evidence-${item.key}`);

            if (!currentTarget) continue;

            currentTarget.innerHTML = `
                <div style="
                    margin-bottom:12px;
                    font-weight:700;
                ">
                    ${item.title}
                </div>

                <img
                    src="${imageURL}"
                    alt="${item.title}"
                    style="
                        display:block;
                        width:100%;
                        max-height:480px;
                        object-fit:contain;
                        background:#09090b;
                        border-radius:10px;
                        border:1px solid #37303f;
                    "
                >
            `;

        } catch (error) {
            const currentTarget = $(`#evidence-${item.key}`);

            if (currentTarget) {
                currentTarget.innerHTML = `
                    <div class="empty">
                        ${esc(error.message)}
                    </div>
                `;
            }
        }
    }
}

/* =========================================================
   CASE DETAILS
========================================================= */

function detailPage() {
    const c = S.cases.find(c => c.case_id === S.caseId);

    if (!c) {
        return `
            <div class="empty">
                Case not found.
                ${btn('Go to Cases', 'allcases')}
            </div>
        `;
    }

    const tabs = [
        'overview',
        'ai',
        'review',
        'task',
        'after',
        'finalize'
    ];

    let body = '';

    /* OVERVIEW */

    if (S.tab === 'overview') {
        body = `
            <div class="grid two">

                ${box('Case Information', detail(c))}

                ${box(
                    'Recommended Action',
                    `
                        <p>${esc(c.recommendation)}</p>

                        <div class="actions">
                            ${btn(
                                'Review / Assign',
                                'detailtab:review',
                                'primary'
                            )}

                            ${btn(
                                'Field Evidence',
                                'detailtab:after'
                            )}
                        </div>

                        <div class="notice" style="margin-top:20px">
                            ${
                                c.verified_cleanup_success
                                    ? 'Verified cleanup'
                                    : 'No independently verified cleanup claim.'
                            }
                        </div>
                    `
                )}

            </div>
        `;
    }

    /* ACTUAL YOLO PREDICTION */

    if (S.tab === 'ai') {
        body = `
            <div class="stack">

                <div class="grid two">

                    ${box(
                        'YOLO F2 Detected Objects',
                        detections(c)
                    )}

                    ${box(
                        'AI Interpretation',
                        `
                            <div class="listrow">
                                <span>Inspection priority</span>
                                ${tag(c.priority?.level)}
                            </div>

                            <div class="listrow">
                                <span>Priority score</span>
                                <b>${esc(c.priority?.score)}</b>
                            </div>

                            <p>
                                ${esc(c.priority?.explanation)}
                            </p>

                            <p style="margin-top:15px">
                                This is an object detector.
                                It does not confirm larvae,
                                Aedes mosquitoes, dengue transmission,
                                or successful cleanup.
                            </p>
                        `
                    )}

                </div>

                <div class="grid two">

                    ${box(
                        'Original Ground Photo',
                        `
                            <div
                                id="evidence-before"
                                class="empty"
                            >
                                Loading original image...
                            </div>
                        `
                    )}

                    ${box(
                        'Actual YOLO Prediction · Bounding Box',
                        `
                            <div
                                id="evidence-before_overlay"
                                class="empty"
                            >
                                Loading YOLO prediction...
                            </div>
                        `
                    )}

                </div>

            </div>
        `;
    }

    /* HUMAN REVIEW */

    if (S.tab === 'review') {
        body = box(
            'Initial Human Review',
            `
                <p>
                    Only cases awaiting initial review
                    can be approved or rejected.
                </p>

                <form
                    id="reviewForm"
                    class="stack"
                    style="margin-top:17px"
                >

                    <div class="field">
                        <label>Reviewer</label>
                        <input
                            name="reviewer"
                            value="Demo Admin"
                            required
                        >
                    </div>

                    <div class="field">
                        <label>Decision</label>

                        <select name="decision">
                            <option value="APPROVE">
                                APPROVE — create field task
                            </option>

                            <option value="REJECT">
                                REJECT — no field task
                            </option>
                        </select>
                    </div>

                    <div class="field">
                        <label>Notes</label>
                        <textarea
                            name="notes"
                            placeholder="Explain why this evidence should or should not proceed"
                        ></textarea>
                    </div>

                    <button
                        class="btn primary"
                        ${c.stage !== 'AWAITING_HUMAN_REVIEW' ? 'disabled' : ''}
                    >
                        Submit Human Review
                    </button>

                </form>

                ${
                    c.stage !== 'AWAITING_HUMAN_REVIEW'
                        ? `
                            <p class="warn" style="margin-top:12px">
                                Current stage: ${esc(c.stage)}.
                                Initial review is not available.
                            </p>
                        `
                        : ''
                }
            `
        );
    }

    /* WORKER TASK */

    if (S.tab === 'task') {
        body = box(
            'Worker Assignment & Task Start',
            `
                <div class="listrow">
                    <span>Current stage</span>
                    ${tag(c.stage)}
                </div>

                <div class="listrow">
                    <span>Assigned worker</span>
                    <b>${esc(c.assigned_worker || '—')}</b>
                </div>

                <form
                    id="assignForm"
                    class="stack"
                    style="margin-top:15px"
                >

                    <div class="field">
                        <label>Worker name</label>

                        <input
                            name="worker"
                            value="${esc(c.assigned_worker || 'worker')}"
                            required
                        >
                    </div>

                    <div class="field">
                        <label>Assignment notes</label>

                        <textarea
                            name="notes"
                            placeholder="Site inspection instructions"
                        ></textarea>
                    </div>

                    <button
                        class="btn primary"
                        ${c.stage !== 'TASK_READY' ? 'disabled' : ''}
                    >
                        Assign Worker
                    </button>

                </form>

                <div class="actions">
                    ${btn(
                        '▶ Start Assigned Task',
                        'starttask',
                        'primary ' +
                        (c.stage === 'TASK_ASSIGNED' ? '' : 'disabled')
                    )}
                </div>

                <p class="smalltext">
                    Assignment requires TASK_READY.
                    Start requires TASK_ASSIGNED.
                </p>
            `
        );
    }

    /* AFTER PHOTO */

    if (S.tab === 'after') {
        body = box(
            'After-Action Evidence',
            `
                <div class="notice">
                    Only upload a genuine photo of the SAME site
                    after an actual action.
                    The checkbox is a worker declaration,
                    not independent verification.
                    Do not reuse the before image.
                </div>

                <div class="listrow">
                    <span>Case stage</span>
                    ${tag(c.stage)}
                </div>

                <form id="afterForm" class="stack">

                    <div class="field">
                        <label>Worker</label>

                        <input
                            name="worker"
                            value="${esc(c.assigned_worker || 'Demo Worker')}"
                            required
                        >
                    </div>

                    <div class="field">
                        <label>After-action photo</label>

                        <input
                            type="file"
                            name="image"
                            accept="image/png,image/jpeg"
                            required
                        >
                    </div>

                    <div class="field">
                        <label>Evidence notes</label>

                        <textarea
                            name="notes"
                            placeholder="Describe what was actually done"
                        ></textarea>
                    </div>

                    <label class="checkrow">
                        <input
                            type="checkbox"
                            name="same_site_attested"
                            required
                        >

                        I attest that this is a genuine
                        after-action photo of the same site.
                    </label>

                    <button
                        class="btn primary"
                        ${c.stage !== 'WORK_IN_PROGRESS' ? 'disabled' : ''}
                    >
                        ⇧ Submit After Evidence
                    </button>

                </form>

                ${
                    c.stage !== 'WORK_IN_PROGRESS'
                        ? `
                            <p class="smalltext warn">
                                After evidence submission
                                requires WORK_IN_PROGRESS.
                            </p>
                        `
                        : ''
                }
            `
        );
    }

    /* FINAL HUMAN REVIEW */

    if (S.tab === 'finalize') {
        body = box(
            'Final Human Evidence Review',
            `
                <div class="listrow">
                    <span>Stage</span>
                    ${tag(c.stage)}
                </div>

                <div class="listrow">
                    <span>AI Change Assessment</span>
                    <b>${esc(c.ai_change_assessment || 'Not assessed')}</b>
                </div>

                <div class="listrow">
                    <span>Image Alignment</span>
                    <b>${esc(c.alignment?.status || 'Not assessed')}</b>
                </div>

                <div class="notice">
                    Final review is an evidence-based human decision.
                    It is NOT independent field verification.
                    Do not mark resolved without sufficient evidence.
                </div>

                <form id="finalForm" class="stack">

                    <div class="field">
                        <label>Final reviewer</label>

                        <input
                            name="reviewer"
                            value="Demo Admin"
                            required
                        >
                    </div>

                    <div class="field">
                        <label>Decision</label>

                        <select name="decision">
                            <option value="REINSPECTION_REQUIRED">
                                REINSPECTION_REQUIRED
                            </option>

                            <option value="UNRESOLVED">
                                UNRESOLVED
                            </option>

                            <option value="RESOLVED_EVIDENCE_REVIEWED">
                                RESOLVED_EVIDENCE_REVIEWED
                            </option>
                        </select>
                    </div>

                    <div class="field">
                        <label>Reviewer notes</label>

                        <textarea
                            name="notes"
                            placeholder="Describe evidence and reasoning"
                        ></textarea>
                    </div>

                    <button
                        class="btn primary"
                        ${c.stage !== 'AWAITING_FINAL_HUMAN_REVIEW' ? 'disabled' : ''}
                    >
                        ☑ Submit Final Review
                    </button>

                </form>

                ${
                    c.stage !== 'AWAITING_FINAL_HUMAN_REVIEW'
                        ? `
                            <p class="warn smalltext">
                                Final review requires a submitted after-action photo.
                            </p>
                        `
                        : ''
                }
            `
        );
    }

    return `
        <div class="hero">

            <div class="kicker">
                CASE DETAIL / LIVE WORKFLOW
            </div>

            <h2>
                ${esc(c.case_id)}
                ${tag(c.priority?.level)}
            </h2>

            <p>
                ${esc(c.stage)}
                ·
                ${esc(c.evidence_mode)}
            </p>

        </div>

        <div class="tabs">
            ${tabs.map(t => `
                <button
                    class="tab ${S.tab === t ? 'active' : ''}"
                    data-tab="${t}"
                >
                    ${
                        {
                            overview: 'Overview',
                            ai: 'AI Detection',
                            review: 'Human Review',
                            task: 'Worker Task',
                            after: 'After Photo',
                            finalize: 'Final Review'
                        }[t]
                    }
                </button>
            `).join('')}
        </div>

        ${body}
    `;
}

/* =========================================================
   MAIN RENDER FUNCTION
========================================================= */

function dronePage() {
    return `
        <div class="panel" style="padding:20px">
            <h2>Drone Mission Simulation</h2>
            <p>Planned surveillance only. No real drone flights or photos.</p>
            <div id="droneMission">Loading mission data...</div>
        </div>
    `;
}

function drawDroneRoute(points) {
    const valid = points.filter(p =>
        Number.isFinite(Number(p.x_m)) &&
        Number.isFinite(Number(p.y_m)) &&
        p.x_m != null && p.y_m != null
    );

    if (!valid.length) {
        return '<p>Route coordinates unavailable.</p>';
    }

    const W = 900, H = 400, pad = 45;
    const xs = valid.map(p => Number(p.x_m));
    const ys = valid.map(p => Number(p.y_m));

    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const minY = Math.min(...ys), maxY = Math.max(...ys);

    const X = v => pad + (Number(v) - minX) /
        (maxX - minX || 1) * (W - pad * 2);

    const Y = v => H - pad - (Number(v) - minY) /
        (maxY - minY || 1) * (H - pad * 2);

    const groups = {};

    valid.forEach(p => {
        const key = `${p.drone}-${p.sortie}`;
        (groups[key] ||= []).push(p);
    });

    const colors = ['#a78bfa', '#22d3ee', '#f59e0b', '#34d399'];

    const routes = Object.entries(groups).map(([key, list], i) => {
        list.sort((a, b) => Number(a.sequence) - Number(b.sequence));

        const color = colors[i % colors.length];

        const line = list.map(p =>
            `${X(p.x_m)},${Y(p.y_m)}`
        ).join(' ');

        const dots = list.map(p => `
            <circle cx="${X(p.x_m)}" cy="${Y(p.y_m)}"
                r="7" fill="${color}" stroke="#111" stroke-width="2"/>
            <text x="${X(p.x_m) + 10}" y="${Y(p.y_m) - 10}"
                fill="white" font-size="12">
                ${Number(p.sequence)}
            </text>
        `).join('');

        return `
            <polyline points="${line}" fill="none"
                stroke="${color}" stroke-width="3"
                stroke-linejoin="round"/>
            ${dots}
        `;
    }).join('');

    return `
        <h3>Planned Drone Route Map</h3>
        <p>Local XY coordinates (meters) — simulated routes, not GPS flights.</p>
        <svg viewBox="0 0 ${W} ${H}"
             style="width:100%;background:#11131c;border-radius:12px">
            ${routes}
        </svg>
        <p style="color:#a78bfa">
            Purple: Drone 1 &nbsp; | &nbsp;
            Cyan: Drone 2
        </p>
    `;
}


async function loadDroneMission() {
    const target = document.querySelector('#droneMission');
    if (!target) return;

    try {
        const data = await api('/drone/mission');
        if (!document.querySelector('#droneMission')) return;

        const summary = data.summary || {};
        const points = data.waypoints || [];

        const escapeHTML = value =>
            String(value ?? '').replace(/[&<>"']/g, ch => ({
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                '"': '&quot;',
                "'": '&#39;'
            }[ch]));

        const rows = points.map(p => `
            <tr>
                <td>${escapeHTML(p.waypoint_id)}</td>
                <td>${escapeHTML(p.drone)}</td>
                <td>${escapeHTML(p.sortie)}</td>
                <td>${escapeHTML(p.sequence)}</td>
                <td>${escapeHTML(p.visit_type)}</td>
            </tr>
        `).join('');

        target.innerHTML = `
        ${drawDroneRoute(points)}
            <h3>Mission: ${escapeHTML(data.mission_id)}</h3>
            <p><strong>Scope:</strong> ${escapeHTML(summary.scope)}</p>
            <p><strong>Drones:</strong> ${escapeHTML(summary.drones)}</p>
            <p><strong>Sorties:</strong> ${escapeHTML(summary.sorties)}</p>
            <p><strong>Waypoints:</strong> ${escapeHTML(points.length)}</p>
            <p><strong>Real Flights:</strong> ${escapeHTML(summary.real_drone_flights)}</p>

            <h3>Planned Waypoints</h3>
            <div style="overflow-x:auto">
                <table style="width:100%;text-align:left">
                    <thead>
                        <tr>
                            <th>Waypoint</th>
                            <th>Drone</th>
                            <th>Sortie</th>
                            <th>Sequence</th>
                            <th>Visit Type</th>
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
        `;
    } catch (e) {
        if (document.querySelector('#droneMission')) {
            target.textContent = 'Mission loading failed: ' + e.message;
        }
    }
}



function render() {
    const titles = {
        dashboard: [
            'Executive Dashboard',
            'AI-assisted breeding-site detection and field response.'
        ],

        report: [
            'New Report & AI Detection',
            'Submit ground evidence for live YOLO inference.'
        ],

        cases: [
            'Case Management',
            'Search, filter and inspect reported cases.'
        ],

       
        map: [
            'GPS Surveillance Map',
            'Location-aware breeding-site inspection and monitoring.'
        ],

        drone: [
            'Drone Mission Simulation',
            'Planned waypoint routes and mission details.'
        ],

        tasks: [
            'Worker Tasks',
            'Assignment and field action workflow.'
        ],



        review: [
            'Human Review',
            'Human oversight for every AI-generated case.'
        ],

        after: [
            'After Evidence & Finalize',
            'Accountable after-action and final human review.'
        ],

        analytics: [
            'Analytics',
            'Descriptive summaries of locally stored cases.'
        ],

        alerts: [
            'Priority Alerts',
            'Case-based review and field follow-up queue.'
        ],

        settings: [
            'Connection & Settings',
            'Connect securely to your own local backend.'
        ],

        detail: [
            'Case Details',
            'Evidence, model outputs and action history.'
        ]
    };

    const [title, sub] = titles[S.page] || titles.dashboard;

    $('#pageTitle').textContent = title;
    $('#subtitle').textContent = sub;
    $('#crumb').textContent = title;

    $('#nav').innerHTML = nav.map(
        ([key, icon, label]) => `
            <button
                class="navitem ${S.page === key ? 'active' : ''}"
                data-page="${key}"
            >
                <span class="ico">${icon}</span>
                ${label}
            </button>
        `
    ).join('');

    

const pages = {
    dashboard,
    report,
    cases,
    map: mapPage,
    drone: dronePage,
    tasks,
    review,
    after,
    analytics,
    alerts,
    settings,
    detail: detailPage
};



    $('#view').innerHTML = pages[S.page]();

    $('#connection').textContent = S.health
        ? 'Online · YOLO'
        : 'Not connected';

    if (S.page === 'detail' && S.tab === 'ai') {
        clearEvidenceUrls();
        loadEvidenceImages();
    }

    if (S.page === 'map') {
        loadGPSMap();
    }
    if (S.page === 'drone') {
    loadDroneMission();
}
}


/* =========================================================
   API MUTATIONS
========================================================= */

async function mutation(path, opts) {
    if (S.busy) return;

    S.busy = true;

    try {
        const c = await api(path, opts);

        toast(
            `Success · ${c.case_id || 'API updated'} · ${c.stage || ''}`
        );

        await sync();

        if (c.case_id) {
            S.lastCase = c;
            S.caseId = c.case_id;
            S.page = 'detail';
            S.tab = 'overview';
            render();
        }

    } catch (e) {
        toast(e.message, true);

    } finally {
        S.busy = false;
    }
}

function formObject(f) {
    return Object.fromEntries(
        new FormData(f).entries()
    );
}

/* =========================================================
   CLICK EVENTS
========================================================= */

document.addEventListener('click', async e => {
    const n = e.target.closest(
        '[data-page],[data-action],[data-case],[data-tab]'
    );

    if (!n) return;

    if (n.dataset.page) {
        go(n.dataset.page);
        return;
    }

    if (n.dataset.case) {
        selectCase(n.dataset.case);
        return;
    }

    if (n.dataset.tab) {
        clearEvidenceUrls();

        S.tab = n.dataset.tab;

        render();
        return;
    }

    const a = n.dataset.action;

    if (a?.startsWith('open:')) {
        return selectCase(a.slice(5));
    }

    if (a?.startsWith('detailtab:')) {
        clearEvidenceUrls();

        S.tab = a.slice(10);

        render();
        return;
    }

    if (a === 'newreport') {
        return go('report');
    }

    if (a === 'allcases') {
        return go('cases');
    }

    if (a === 'openmap') {
        return go('map');
    }

    if (a === 'refreshmap') {
        return loadGPSMap();
    }

    if (a === 'pendingreview') {
        return go('review');
    }

    if (a === 'workertasks') {
        return go('tasks');
    }

    if (a === 'afterpage') {
        return go('after');
    }

    if (a === 'lastcase' && S.lastCase) {
        return selectCase(S.lastCase.case_id);
    }

    if (a === 'disconnect') {
        clearEvidenceUrls();

        S.token = '';
        S.health = null;
        S.cases = [];
        S.dashboard = null;

        toast('Disconnected. Token cleared from this tab.');

        render();
        return;
    }

    
});

/* =========================================================
   INPUT EVENTS
========================================================= */

document.addEventListener('input', e => {

    if (e.target.id === 'caseSearch') {
        S.search = e.target.value;

        const cs = S.cases.filter(c =>
            (
                !S.search ||
                JSON.stringify([
                    c.case_id,
                    c.stage,
                    c.assigned_worker,
                    c.f2_detections?.[0]?.class_name
                ]).toLowerCase().includes(S.search.toLowerCase())
            ) &&
            (
                S.priority === 'ALL' ||
                c.priority?.level === S.priority
            ) &&
            (
                S.stage === 'ALL' ||
                c.stage === S.stage
            )
        );

        const results = $('#caseResults');

        if (results) {
            results.innerHTML = caseTable(cs);
        }
    }

    if (
        e.target.id === 'photo' &&
        e.target.files?.[0]
    ) {
        const f = e.target.files[0];

        const p = $('#uploadPreview');

        if (!p) return;

        const url = URL.createObjectURL(f);

        p.innerHTML = `
            <img
                alt="Selected site image preview"
                src="${url}"
            >
        `;

        p.dataset.url = url;
    }
});

/* =========================================================
   FILTER EVENTS
========================================================= */

document.addEventListener('change', e => {

    if (e.target.id === 'priorityFilter') {
        S.priority = e.target.value;
        render();
    }

    if (e.target.id === 'stageFilter') {
        S.stage = e.target.value;
        render();
    }
});

/* =========================================================
   FORM SUBMISSION
========================================================= */

document.addEventListener('submit', async e => {

    const f = e.target;

    if (![
        'settingsForm',
        'reportForm',
        'reviewForm',
        'assignForm',
        'finalForm'
    ].includes(f.id)) {
        return;
    }

    e.preventDefault();

    /* SETTINGS */

    if (f.id === 'settingsForm') {

        const base = $('#apiBase')
            .value
            .trim()
            .replace(/\/$/, '');


        
        if (!/^https?:\/\/(127\.0\.0\.1|localhost):8003$/.test(base)) {
           return toast(
               'For this local demo, use http://127.0.0.1:8003.',
               true
            );
        }



        S.base = base;
        S.token = $('#apiToken').value.trim();

        if (!S.token) {
            return toast('Enter your Bearer token.', true);
        }

        await sync();
        return;
    }

    /* NEW REPORT + REAL YOLO */

    if (f.id === 'reportForm') {

        const fd = new FormData(f);

        const file = fd.get('image');

        if (!file?.size) {
            return toast('Select an image.', true);
        }

        if (file.size > 12 * 1024 * 1024) {
            return toast('Image must be under 12 MB.', true);
        }

        if (
            Boolean(fd.get('latitude')) !==
            Boolean(fd.get('longitude'))
        ) {
            return toast(
                'Provide both GPS coordinates or neither.',
                true
            );
        }

        if (!fd.get('latitude')) {
            fd.delete('latitude');
        }

        if (!fd.get('longitude')) {
            fd.delete('longitude');
        }

        return mutation(
            '/cases',
            {
                method: 'POST',
                body: fd
            }
        );
    }

    if (!S.caseId) {
        return toast('Select a case first.', true);
    }

    let path;
    let body;

    /* AFTER-ACTION IMAGE */

    if (f.id === 'afterForm') {

        const fd = new FormData(f);

        const file = fd.get('image');

        if (!file?.size) {
            return toast(
                'Select an after-action image.',
                true
            );
        }

        if (!f.querySelector('[name=same_site_attested]').checked) {
            return toast(
                'Same-site attestation required.',
                true
            );
        }

        fd.set('same_site_attested', 'true');

        return mutation(
            `/cases/${encodeURIComponent(S.caseId)}/after`,
            {
                method: 'POST',
                body: fd
            }
        );
    }

    /* REVIEW / ASSIGN / FINALIZE */

    body = formObject(f);

    path = ({
        reviewForm: 'review',
        assignForm: 'assign',
        finalForm: 'finalize'
    })[f.id];

    return mutation(
        `/cases/${encodeURIComponent(S.caseId)}/${path}`,
        {
            method: 'POST',
            body: JSON.stringify(body)
        }
    );
});

/* =========================================================
   REFRESH
========================================================= */

$('#refresh').addEventListener('click', sync);

/* INITIAL RENDER */






/* INITIAL RENDER — ADMIN SESSION */

(async function initializeAdminDashboard() {
    const raw =
        sessionStorage.getItem('dengu_admin_session') ||
        sessionStorage.getItem('dengu_admin_handoff');

    if (!raw) {
        window.location.replace('roles.html');
        return;
    }

    try {
        const login = JSON.parse(raw);

        const token = login.token || login.access_token;

        if (!token) {
            throw new Error('Admin session token missing');
        }

        if (login.role && login.role !== 'admin') {
            throw new Error('Admin role required');
        }

        S.base = 'http://127.0.0.1:8003';
        S.token = token;

        // Verify the session and role with the backend.
        const me = await api('/auth/me');

        if (!me || me.role !== 'admin') {
            throw new Error('Admin access required');
        }

        const profile = document.querySelector('.profile');

        if (profile) {
            profile.textContent =
                '● ' + (login.username || 'Admin') + ' ▾';
        }

        render();
        await sync();

    } catch (error) {
        console.error('Dashboard initialization failed:', error);
        S.token = '';

        sessionStorage.removeItem('dengu_admin_session');
        sessionStorage.removeItem('dengu_admin_handoff');

        window.location.replace('roles.html');
    }
})();

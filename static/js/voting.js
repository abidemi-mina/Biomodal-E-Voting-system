/**
 * NigeriaVotes — Shared JavaScript Utilities
 * ============================================
 * Used across all voting pages for common functionality.
 */

'use strict';

// ── CSRF helper ───────────────────────────────────────────────
function getCsrf() {
    return document.cookie
        .split(';')
        .find(c => c.trim().startsWith('csrftoken='))
        ?.split('=')[1] || '';
}

// ── Fetch wrapper with CSRF ───────────────────────────────────
async function apiFetch(url, data) {
    const resp = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCsrf(),
        },
        body: JSON.stringify(data),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return resp.json();
}

// ── Camera Manager ────────────────────────────────────────────
class CameraManager {
    constructor(videoEl, canvasEl, options = {}) {
        this.video = typeof videoEl === 'string' ? document.getElementById(videoEl) : videoEl;
        this.canvas = typeof canvasEl === 'string' ? document.getElementById(canvasEl) : canvasEl;
        this.stream = null;
        this.mirror = options.mirror !== false;  // default: mirror (selfie cam)
        this.quality = options.quality || 0.85;
        this.onReady = options.onReady || null;
        this.onError = options.onError || null;
    }

    async start() {
        try {
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user',
                },
                audio: false,
            });
            this.video.srcObject = this.stream;
            return new Promise((resolve) => {
                this.video.onloadedmetadata = () => {
                    this.canvas.width = this.video.videoWidth;
                    this.canvas.height = this.video.videoHeight;
                    if (this.onReady) this.onReady();
                    resolve(true);
                };
            });
        } catch (err) {
            console.error('Camera error:', err);
            if (this.onError) this.onError(err);
            return false;
        }
    }

    capture() {
        const ctx = this.canvas.getContext('2d');
        if (this.mirror) {
            ctx.save();
            ctx.scale(-1, 1);
            ctx.drawImage(this.video, -this.canvas.width, 0, this.canvas.width, this.canvas.height);
            ctx.restore();
        } else {
            ctx.drawImage(this.video, 0, 0);
        }
        return this.canvas.toDataURL('image/jpeg', this.quality);
    }

    stop() {
        if (this.stream) {
            this.stream.getTracks().forEach(t => t.stop());
            this.stream = null;
        }
    }

    get isActive() {
        return this.stream !== null;
    }
}

// ── Flash effect on camera ────────────────────────────────────
function cameraFlash(container) {
    const el = document.createElement('div');
    el.style.cssText = `
        position: absolute; inset: 0;
        background: white; opacity: .7;
        pointer-events: none;
        border-radius: inherit;
        animation: nvFlash .35s ease forwards;
    `;
    if (!document.querySelector('#nv-flash-style')) {
        const s = document.createElement('style');
        s.id = 'nv-flash-style';
        s.textContent = '@keyframes nvFlash { 0% {opacity:.7} 100% {opacity:0} }';
        document.head.appendChild(s);
    }
    container.appendChild(el);
    setTimeout(() => el.remove(), 400);
}

// ── Progress / loading helpers ────────────────────────────────
function showLoader(btnEl, text = 'Processing…') {
    btnEl._originalText = btnEl.innerHTML;
    btnEl.disabled = true;
    btnEl.innerHTML = `<span style="width:16px;height:16px;border:2px solid rgba(255,255,255,.3);border-top-color:white;border-radius:50%;display:inline-block;animation:nvSpin .6s linear infinite;"></span> ${text}`;
    if (!document.querySelector('#nv-spin-style')) {
        const s = document.createElement('style');
        s.id = 'nv-spin-style';
        s.textContent = '@keyframes nvSpin { to { transform: rotate(360deg); } }';
        document.head.appendChild(s);
    }
}

function hideLoader(btnEl) {
    if (btnEl._originalText) {
        btnEl.innerHTML = btnEl._originalText;
    }
    btnEl.disabled = false;
}

// ── Alert helpers ─────────────────────────────────────────────
function showAlert(containerId, type, message) {
    const el = typeof containerId === 'string' ? document.getElementById(containerId) : containerId;
    if (!el) return;
    const icons = { success: '✅', error: '❌', warning: '⚠️', info: 'ℹ️' };
    el.className = `alert alert-${type}`;
    el.innerHTML = `<span>${icons[type] || 'ℹ️'}</span><span>${message}</span>`;
    el.style.display = 'flex';
}

function hideAlert(containerId) {
    const el = typeof containerId === 'string' ? document.getElementById(containerId) : containerId;
    if (el) el.style.display = 'none';
}

// ── Countdown timer ───────────────────────────────────────────
function startCountdown(seconds, onTick, onEnd) {
    let remaining = seconds;
    onTick(remaining);
    const interval = setInterval(() => {
        remaining--;
        onTick(remaining);
        if (remaining <= 0) {
            clearInterval(interval);
            if (onEnd) onEnd();
        }
    }, 1000);
    return interval;
}

// ── Format score as percentage ────────────────────────────────
function formatScore(score, decimals = 1) {
    return (score * 100).toFixed(decimals) + '%';
}

// ── Export globals ────────────────────────────────────────────
window.NV = {
    getCsrf,
    apiFetch,
    CameraManager,
    cameraFlash,
    showLoader,
    hideLoader,
    showAlert,
    hideAlert,
    startCountdown,
    formatScore,
};

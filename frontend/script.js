/**
 * UK House Price Predictor - Frontend Logic (Fixed)
 * Handles form submission, validation, API communication, and result rendering.
 */

// Use same-origin first, then fallback to localhost for local Flask development.
const API_BASE =
    window.location.origin.startsWith('http')
        ? window.location.origin
        : 'http://localhost:5000';

// DOM Elements
let form;
let predictBtn;
let resetBtn;
let resultDiv;
let resultMeta;
let messageDiv;
let apiStatus;

// Prevent duplicate requests
let isPredicting = false;

// Helper: safely get element by id
function getEl(id) {
    return document.getElementById(id);
}

// Helper: Display message
function showMessage(text, type = 'info') {
    if (!messageDiv) return;
    messageDiv.textContent = text;
    messageDiv.classList.remove('hidden', 'error', 'success', 'info');
    if (type === 'error') messageDiv.classList.add('error');
    else if (type === 'success') messageDiv.classList.add('success');
    else messageDiv.classList.add('info');
}

function hideMessage() {
    if (!messageDiv) return;
    messageDiv.classList.add('hidden');
}

// Helper: Update API status badge
function setApiStatus(status, text) {
    if (!apiStatus) return;
    apiStatus.className = `status-badge ${status}`;
    apiStatus.textContent = text;
}

// Helper: safely set result
function setResult(value, meta = '') {
    if (resultDiv) resultDiv.textContent = value;
    if (resultMeta) resultMeta.textContent = meta;
}

// Check API health on page load
async function checkApiHealth() {
    try {
        const response = await fetch(`${API_BASE}/`, {
            method: 'GET',
            headers: { 'Accept': 'application/json, text/plain, */*' }
        });

        if (response.ok) {
            setApiStatus('success', 'API Connected');
        } else {
            setApiStatus('warning', `API Error (${response.status})`);
        }
    } catch (error) {
        setApiStatus('error', 'API Offline');
        console.warn('API health check failed:', error);
    }
}

// Reset form to default values
function resetForm() {
    const defaults = {
        property_type: 'D',
        new_build: '0',
        freehold: '1',
        prediction_date: '',
        Town: 'London',
        County: 'Greater London',
        avg_1m_change: '0.4',
        avg_12m_change: '4.2',
        salesvolume: '850',
        detachedprice: '720000',
        detached1mpctchange: '0.3',
        detached12mpctchange: '4.8',
        semidetachedprice: '510000',
        semidetached1mpctchange: '0.4',
        semidetached12mpctchange: '4.5',
        terracedprice: '430000',
        terraced1mpctchange: '0.5',
        terraced12mpctchange: '4.1',
        flatprice: '370000',
        flat1mpctchange: '0.2',
        flat12mpctchange: '3.6'
    };

    Object.entries(defaults).forEach(([id, value]) => {
        const el = getEl(id);
        if (el) el.value = value;
    });

    // set today's date after reset
    const dateEl = getEl('prediction_date');
    if (dateEl) {
        dateEl.value = new Date().toISOString().split('T')[0];
    }

    setResult('£ ---', 'Awaiting prediction');
    hideMessage();
    setApiStatus('success', 'Ready');
}

// Parse number safely
function toNumber(id, fallback = 0) {
    const el = getEl(id);
    if (!el) return fallback;
    const value = parseFloat(el.value);
    return Number.isFinite(value) ? value : fallback;
}

// Parse integer safely
function toInt(id, fallback = 0) {
    const el = getEl(id);
    if (!el) return fallback;
    const value = parseInt(el.value, 10);
    return Number.isFinite(value) ? value : fallback;
}

// Build payload expected by backend/model
function buildPayload() {
    const propType = getEl('property_type')?.value || 'D';

    const typeFeatures = {
        type_D: propType === 'D' ? 1 : 0,
        type_S: propType === 'S' ? 1 : 0,
        type_T: propType === 'T' ? 1 : 0,
        type_F: propType === 'F' ? 1 : 0
    };

    const dateInput = getEl('prediction_date')?.value;
    let year = 2024;
    let month = 1;
    let quarter = 1;

    if (dateInput) {
        const d = new Date(dateInput);
        if (!Number.isNaN(d.getTime())) {
            year = d.getFullYear();
            month = d.getMonth() + 1;
            quarter = Math.floor(d.getMonth() / 3) + 1;
        }
    }

    return {
        ...typeFeatures,

        new_build: toInt('new_build', 0),
        freehold: toInt('freehold', 1),

        year,
        month,
        quarter,

        Town: (getEl('Town')?.value || '').trim(),
        County: (getEl('County')?.value || '').trim(),

        avg_1m_change: toNumber('avg_1m_change'),
        avg_12m_change: toNumber('avg_12m_change'),
        salesvolume: toNumber('salesvolume'),

        detachedprice: toNumber('detachedprice'),
        detached1mpctchange: toNumber('detached1mpctchange'),
        detached12mpctchange: toNumber('detached12mpctchange'),

        semidetachedprice: toNumber('semidetachedprice'),
        semidetached1mpctchange: toNumber('semidetached1mpctchange'),
        semidetached12mpctchange: toNumber('semidetached12mpctchange'),

        terracedprice: toNumber('terracedprice'),
        terraced1mpctchange: toNumber('terraced1mpctchange'),
        terraced12mpctchange: toNumber('terraced12mpctchange'),

        flatprice: toNumber('flatprice'),
        flat1mpctchange: toNumber('flat1mpctchange'),
        flat12mpctchange: toNumber('flat12mpctchange')
    };
}

// Validate payload before sending
function validatePayload(payload) {
    if (!payload.Town) {
        throw new Error('Town is required.');
    }
    if (!payload.County) {
        throw new Error('County is required.');
    }
}

// Read server response safely even if not JSON
async function parseResponse(response) {
    const contentType = response.headers.get('content-type') || '';

    if (contentType.includes('application/json')) {
        return await response.json();
    }

    const text = await response.text();
    return { error: text || `HTTP ${response.status}` };
}

// Prediction handler
async function handlePredict(event) {
    if (event) event.preventDefault();

    if (isPredicting) return;
    isPredicting = true;

    if (predictBtn) {
        predictBtn.disabled = true;
        predictBtn.innerHTML = '<span class="loading-spinner">⟳</span> Predicting...';
    }

    hideMessage();
    setResult('£ ---', 'Calculating...');

    try {
        const payload = buildPayload();
        validatePayload(payload);

        console.log('Sending payload:', payload);

        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json, text/plain, */*'
            },
            body: JSON.stringify(payload)
        });

        const data = await parseResponse(response);

        if (!response.ok) {
            throw new Error(data.error || `HTTP ${response.status}`);
        }

        const predicted =
            data.predicted_price ??
            data.prediction ??
            data.price;

        if (predicted === undefined || predicted === null || Number.isNaN(Number(predicted))) {
            throw new Error('Invalid response from server: predicted price missing.');
        }

        const formattedPrice = new Intl.NumberFormat('en-GB', {
            style: 'currency',
            currency: 'GBP',
            maximumFractionDigits: 0
        }).format(Number(predicted));

        setResult(
            formattedPrice,
            `Prediction based on ${payload.Town}, ${payload.County}`
        );

        showMessage('Prediction successful!', 'success');
        setApiStatus('success', 'API Connected');
    } catch (error) {
        console.error('Prediction error:', error);
        setResult('Error', error.message || 'Prediction failed');
        showMessage(`Error: ${error.message || 'Prediction failed'}`, 'error');
        setApiStatus('error', 'Prediction Failed');
    } finally {
        isPredicting = false;

        if (predictBtn) {
            predictBtn.disabled = false;
            predictBtn.innerHTML = 'Predict Price';
        }
    }
}

// Init app after DOM is ready
function init() {
    form = getEl('prediction-form');
    predictBtn = getEl('predict-btn');
    resetBtn = getEl('reset-btn');
    resultDiv = getEl('result');
    resultMeta = getEl('resultMeta');
    messageDiv = getEl('form-message');
    apiStatus = getEl('api-status');

    if (!form) {
        console.error('Form with id "prediction-form" not found.');
        return;
    }

    if (!predictBtn) {
        console.error('Predict button with id "predict-btn" not found.');
        return;
    }

    // Make sure button behaves properly even if HTML forgot type="submit"
    predictBtn.setAttribute('type', 'submit');

    const dateEl = getEl('prediction_date');
    if (dateEl && !dateEl.value) {
        dateEl.value = new Date().toISOString().split('T')[0];
    }

    form.addEventListener('submit', handlePredict);
    predictBtn.addEventListener('click', handlePredict);

    if (resetBtn) {
        resetBtn.addEventListener('click', function (e) {
            e.preventDefault();
            resetForm();
        });
    }

    checkApiHealth();
    console.log('Frontend initialized successfully');
}

document.addEventListener('DOMContentLoaded', init);
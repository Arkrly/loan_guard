// ===== Configuration =====
const API_BASE = window.location.origin;
const API_URL = `${API_BASE}/predict`;
const HEALTH_URL = `${API_BASE}/health`;

// ===== DOM Elements =====
let loadingScreen, mainApp, progressBar, progressPercent, loadingStatus;
let form, submitBtn, btnContent, btnLoading, historyList, clearHistoryBtn, toastContainer;
let resultPlaceholder, resultContent, resultBadge, resultIcon, resultText;
let scoreCircle, scoreValue, probValue, thresholdValue, resultModelVersion, modelVersion;

// ===== Initialize DOM References =====
function initDOMElements() {
    loadingScreen = document.getElementById('loadingScreen');
    mainApp = document.getElementById('mainApp');
    progressBar = document.getElementById('progressBar');
    progressPercent = document.getElementById('progressPercent');
    loadingStatus = document.getElementById('loadingStatus');
    form = document.getElementById('loanForm');
    submitBtn = document.getElementById('submitBtn');
    btnContent = document.getElementById('btnContent');
    btnLoading = document.getElementById('btnLoading');
    historyList = document.getElementById('historyList');
    clearHistoryBtn = document.getElementById('clearHistoryBtn');
    toastContainer = document.getElementById('toastContainer');

    // Result elements
    resultPlaceholder = document.getElementById('resultPlaceholder');
    resultContent = document.getElementById('resultContent');
    resultBadge = document.getElementById('resultBadge');
    resultIcon = document.getElementById('resultIcon');
    resultText = document.getElementById('resultText');
    scoreCircle = document.getElementById('scoreCircle');
    scoreValue = document.getElementById('scoreValue');
    probValue = document.getElementById('probValue');
    thresholdValue = document.getElementById('thresholdValue');
    resultModelVersion = document.getElementById('resultModelVersion');
    modelVersion = document.getElementById('modelVersion');
}

// ===== Loading Animation =====
async function initializeApp() {
    console.log('Initializing app...');
    let progress = 0;

    // Animate progress
    const progressInterval = setInterval(() => {
        if (progress < 30) {
            progress += 2;
            updateProgress(progress);
        }
    }, 50);

    // Check API health
    try {
        if (loadingStatus) loadingStatus.textContent = 'Connecting to API';
        console.log('Fetching health endpoint:', HEALTH_URL);
        await new Promise(resolve => setTimeout(resolve, 500));

        const response = await fetch(HEALTH_URL);
        console.log('Health response status:', response.status);

        if (response.ok) {
            const data = await response.json();
            console.log('Health data:', data);

            // Update progress
            progress = 50;
            updateProgress(progress);
            if (loadingStatus) loadingStatus.textContent = 'Loading Model';

            await new Promise(resolve => setTimeout(resolve, 300));

            if (data.model_loaded) {
                progress = 75;
                updateProgress(progress);
                if (loadingStatus) loadingStatus.textContent = 'Initializing Interface';

                // Update model version in sidebar
                if (modelVersion) {
                    modelVersion.textContent = data.model_version || '1.0';
                }
            }

            await new Promise(resolve => setTimeout(resolve, 300));
            progress = 100;
            updateProgress(progress);
            if (loadingStatus) loadingStatus.textContent = 'Ready';

            clearInterval(progressInterval);

            // Fade out loading screen
            await new Promise(resolve => setTimeout(resolve, 500));
            showMainApp();

            // Load history
            loadHistory();

        } else {
            throw new Error('API not healthy: ' + response.status);
        }
    } catch (error) {
        clearInterval(progressInterval);
        console.error('Initialization error:', error);

        // Show main app anyway after a delay
        if (loadingStatus) loadingStatus.textContent = 'Starting in offline mode...';
        await new Promise(resolve => setTimeout(resolve, 1500));
        showMainApp();
        loadHistory();
    }
}

// ===== Show Main App =====
function showMainApp() {
    console.log('Showing main app...');
    if (loadingScreen) loadingScreen.classList.add('hidden');
    if (mainApp) mainApp.style.opacity = '1';
}

// ===== Update Progress =====
function updateProgress(percent) {
    if (progressBar) progressBar.style.width = `${percent}%`;
    if (progressPercent) progressPercent.textContent = `${percent}%`;
}

// ===== Initialize on Load =====
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing...');
    initDOMElements();
    
    // Initialize term display first to show correct values
    updateTermDisplay();
    setTermPreset(360); // Set default to 30 years
    
    initializeApp();
    initFormHandler();
});

// ===== Self Employed Toggle =====
let isSelfEmployed = false;

function toggleSelfEmployed() {
    isSelfEmployed = !isSelfEmployed;

    const btn = document.getElementById('selfEmployedBtn');
    const indicator = document.getElementById('selfEmployedIndicator');
    const dot = document.getElementById('selfEmployedDot');
    const input = document.getElementById('selfEmployedInput');
    const textSpan = btn?.querySelector('.text-sm');

    if (isSelfEmployed) {
        // Active state
        btn?.classList.remove('text-slate-400', 'border-white/10');
        btn?.classList.add('text-primary', 'border-primary/30', 'bg-primary/10');
        indicator?.classList.remove('bg-slate-700');
        indicator?.classList.add('bg-primary');
        dot.style.transform = 'translateX(12px)';
        if (textSpan) textSpan.textContent = 'Yes';
        if (input) input.value = 'Yes';
    } else {
        // Inactive state
        btn?.classList.add('text-slate-400', 'border-white/10');
        btn?.classList.remove('text-primary', 'border-primary/30', 'bg-primary/10');
        indicator?.classList.add('bg-slate-700');
        indicator?.classList.remove('bg-primary');
        dot.style.transform = 'translateX(0)';
        if (textSpan) textSpan.textContent = 'No';
        if (input) input.value = 'No';
    }
}

// ===== Loan Term Slider =====
let currentTermMonths = 360;
const MIN_TERM = 12;
const MAX_TERM = 480;

function updateTermFromSlider(value) {
    currentTermMonths = parseInt(value);
    updateTermDisplay();
}

function setTermPreset(months) {
    currentTermMonths = months;
    const slider = document.getElementById('termSlider');
    if (slider) slider.value = months;
    updateTermDisplay();

    // Update preset button styles
    document.querySelectorAll('[onclick^="setTermPreset"]').forEach(btn => {
        const btnMonths = parseInt(btn.getAttribute('onclick').match(/\d+/)[0]);
        if (btnMonths === months) {
            btn.className = 'flex-1 py-2 px-2 text-xs font-medium rounded-lg bg-primary/20 border border-primary/30 text-primary hover:bg-primary/30 transition-all';
        } else {
            btn.className = 'flex-1 py-2 px-2 text-xs font-medium rounded-lg bg-white/5 border border-white/10 text-slate-400 hover:text-white hover:bg-primary/20 hover:border-primary/30 transition-all';
        }
    });
}

function updateTermDisplay() {
    const years = Math.floor(currentTermMonths / 12);
    const months = currentTermMonths % 12;

    const termYearsEl = document.getElementById('termYears');
    const termMonthsEl = document.getElementById('termMonths');
    const termTotalEl = document.getElementById('termTotal');
    const termInputEl = document.getElementById('loanTermInput');

    if (termYearsEl) termYearsEl.textContent = years;
    if (termMonthsEl) termMonthsEl.textContent = months;
    if (termTotalEl) termTotalEl.textContent = currentTermMonths;
    if (termInputEl) termInputEl.value = currentTermMonths;
}

// ===== Toast Notifications =====
function showToast(type, title, message, duration = 5000) {
    const icons = {
        success: 'check_circle',
        error: 'error',
        warning: 'warning'
    };

    const colors = {
        success: 'secondary',
        error: 'danger',
        warning: 'warning'
    };

    const toast = document.createElement('div');
    toast.className = `flex items-start gap-3 p-4 rounded-xl bg-surface-glass backdrop-blur-md border border-white/10 shadow-xl max-w-sm animate-slide-up`;
    toast.innerHTML = `
        <span class="material-symbols-outlined text-${colors[type]}">${icons[type]}</span>
        <div class="flex-1">
            <h4 class="text-white font-semibold text-sm">${title}</h4>
            <p class="text-slate-400 text-xs mt-1">${message}</p>
        </div>
        <button onclick="this.parentElement.remove()" class="text-slate-500 hover:text-white">
            <span class="material-symbols-outlined text-lg">close</span>
        </button>
    `;

    toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(100%)';
        toast.style.transition = 'all 0.3s ease';
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ===== Form Submission =====
function initFormHandler() {
    if (!form) {
        console.error('Form element not found');
        return;
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        console.log('Form submitted');

        // Show loading state
        setLoading(true);

        try {
            const formData = new FormData(form);
            const payload = {
                Gender: formData.get('Gender'),
                Married: formData.get('Married'),
                Dependents: formData.get('Dependents'),
                Education: formData.get('Education'),
                Self_Employed: formData.get('Self_Employed'),
                ApplicantIncome: Number(formData.get('ApplicantIncome')) || 0,
                CoapplicantIncome: Number(formData.get('CoapplicantIncome')) || 0,
                LoanAmount: Number(formData.get('LoanAmount')) || 0,
                Loan_Amount_Term: Number(formData.get('Loan_Amount_Term')),
                Credit_History: Number(formData.get('Credit_History')),
                Property_Area: formData.get('Property_Area')
            };

            // Validate
            if (payload.ApplicantIncome + payload.CoapplicantIncome <= 0) {
                showToast('warning', 'Validation Error', 'Total income must be greater than 0');
                setLoading(false);
                return;
            }

            if (payload.LoanAmount <= 0) {
                showToast('warning', 'Validation Error', 'Loan amount must be greater than 0');
                setLoading(false);
                return;
            }

            // Make API call
            const response = await fetch(API_URL, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                let errorDetail = 'Unknown error';
                try {
                    const err = await response.json();
                    errorDetail = err.detail?.[0]?.msg || err.detail || JSON.stringify(err);
                } catch {
                    errorDetail = response.statusText;
                }
                throw new Error(errorDetail);
            }

            const result = await response.json();

            // Display result
            displayResult(result);

            // Save to history
            saveToHistory(payload, result);

            // Show success toast
            showToast('success', 'Analysis Complete',
                `Loan ${result.prediction.toLowerCase()}. Risk: ${result.risk_score.toFixed(1)}%`);

        } catch (error) {
            console.error('Error:', error);
            showToast('error', 'Analysis Failed', error.message || 'Please check your inputs and try again');
        } finally {
            setLoading(false);
        }
    });

    // ===== Loading State =====
    function setLoading(loading) {
        if (loading) {
            submitBtn.disabled = true;
            btnContent.classList.add('hidden');
            btnLoading.classList.remove('hidden');
            btnLoading.classList.add('flex');
        } else {
            submitBtn.disabled = false;
            btnContent.classList.remove('hidden');
            btnLoading.classList.add('hidden');
            btnLoading.classList.remove('flex');
        }
    }

    // ===== Display Result =====
    function displayResult(data) {
        // Hide placeholder, show result
        if (resultPlaceholder) resultPlaceholder.classList.add('hidden');
        if (resultContent) resultContent.classList.remove('hidden');

        const isApproved = data.prediction === 'Approved';

        // Update badge - modern horizontal chip
        if (resultBadge) {
            resultBadge.className = `flex items-center justify-center gap-3 py-3 px-4 rounded-xl mb-5 transition-all ${isApproved ? 'bg-secondary/20 border border-secondary/30' : 'bg-danger/20 border border-danger/30'}`;
        }
        if (resultIcon) {
            resultIcon.textContent = isApproved ? 'check_circle' : 'cancel';
            resultIcon.className = `material-symbols-outlined text-2xl ${isApproved ? 'text-secondary' : 'text-danger'}`;
        }
        if (resultText) {
            resultText.textContent = isApproved ? 'APPROVED' : 'REJECTED';
            resultText.className = `text-lg font-bold tracking-wide ${isApproved ? 'text-secondary' : 'text-danger'}`;
        }

        // Animate score ring (new circumference: 2 * PI * 42 = 264)
        const score = Math.round(data.risk_score);
        const circumference = 264;
        const offset = circumference - (score / 100) * circumference;

        if (scoreCircle) {
            scoreCircle.style.stroke = isApproved ? '#10b981' : '#ef4444';
            scoreCircle.style.strokeDashoffset = circumference;

            // Animate after brief delay
            setTimeout(() => {
                scoreCircle.style.strokeDashoffset = offset;
                if (scoreValue) animateNumber(scoreValue, 0, score, 1000);
            }, 100);
        }

        // Update stats
        if (probValue) probValue.textContent = `${(data.probability_approved * 100).toFixed(1)}%`;
        if (thresholdValue) thresholdValue.textContent = `${(data.threshold_used * 100).toFixed(1)}%`;
        if (resultModelVersion) resultModelVersion.textContent = `v${data.model_version}`;
    }

    // ===== Animate Number =====
    function animateNumber(element, start, end, duration) {
        const startTime = performance.now();

        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const easeProgress = 1 - Math.pow(1 - progress, 3);
            const current = Math.round(start + (end - start) * easeProgress);
            element.textContent = current;

            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }

        requestAnimationFrame(update);
    }

    // ===== History Management =====
    function saveToHistory(input, output) {
        const history = JSON.parse(localStorage.getItem('loanHistory') || '[]');

        const item = {
            id: `LG-${Date.now().toString().slice(-4)}`,
            timestamp: new Date().toISOString(),
            amount: input.LoanAmount,
            result: output.prediction,
            risk: Math.round(output.risk_score),
            fullInput: input
        };

        history.unshift(item);
        if (history.length > 10) history.pop();

        localStorage.setItem('loanHistory', JSON.stringify(history));
        loadHistory();
    }

    function loadHistory() {
        const history = JSON.parse(localStorage.getItem('loanHistory') || '[]');

        if (history.length === 0) {
            historyList.innerHTML = `
            <div class="text-center py-8 text-slate-500 text-sm">
                <span class="material-symbols-outlined text-3xl mb-2 block opacity-50">folder_open</span>
                No history yet
            </div>
        `;
            return;
        }

        historyList.innerHTML = history.map(item => {
            const date = new Date(item.timestamp);
            const dateStr = date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
            const timeStr = date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });

            const riskClass = item.result === 'Approved'
                ? 'bg-green-500/20 text-green-400 border-green-500/20'
                : 'bg-red-500/20 text-red-400 border-red-500/20';
            const riskLabel = item.result === 'Approved' ? 'LOW RISK' : 'HIGH RISK';

            return `
            <button class="flex items-center justify-between w-full p-3 rounded-lg bg-white/5 hover:bg-white/10 border border-white/5 transition-all group">
                <div class="flex flex-col items-start gap-0.5">
                    <span class="text-sm font-medium text-white">#${item.id}</span>
                    <span class="text-xs text-slate-400">${dateStr}, ${timeStr}</span>
                </div>
                <span class="px-2 py-1 rounded text-[10px] font-bold ${riskClass} border">${riskLabel}</span>
            </button>
        `;
        }).join('');
    }

    // ===== Clear History =====
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', () => {
            localStorage.removeItem('loanHistory');
            loadHistory();

            // Reset result panel
            if (resultPlaceholder) resultPlaceholder.classList.remove('hidden');
            if (resultContent) resultContent.classList.add('hidden');

            showToast('success', 'History Cleared', 'All previous analyses have been removed');
        });
    }
}

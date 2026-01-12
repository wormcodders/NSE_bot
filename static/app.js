/**
 * Stock Signal Dashboard - Application JavaScript
 * Handles frontend logic for signals display and depth analysis
 */

class StockSignalApp {
    constructor() {
        this.currentTab = 'signals';
        this.signalsPollInterval = null;
        this.analysisPollInterval = null;
        this.lastSignalsTimestamp = null;
        
        this.init();
    }
    
    init() {
        this.bindEvents();
        this.loadInitialData();
    }
    
    bindEvents() {
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const tab = e.currentTarget.dataset.tab;
                this.switchTab(tab);
            });
        });
        
        // Symbol input
        const symbolInput = document.getElementById('symbolInput');
        symbolInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.startDepthAnalysis();
            }
        });
        
        // Analyze button
        document.getElementById('analyzeBtn').addEventListener('click', () => {
            this.startDepthAnalysis();
        });
        
        // Retry button
        document.getElementById('retryBtn').addEventListener('click', () => {
            this.startDepthAnalysis();
        });
    }
    
    switchTab(tab) {
        this.currentTab = tab;
        
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tab);
        });
        
        // Update tab panels
        document.querySelectorAll('.tab-panel').forEach(panel => {
            panel.classList.toggle('active', panel.id === `${tab}-panel`);
        });
        
        // Handle polling based on active tab
        if (tab === 'signals') {
            this.startSignalsPolling();
            this.stopAnalysisPolling();
        } else {
            this.stopSignalsPolling();
        }
    }
    
    loadInitialData() {
        // Load signals immediately
        this.loadSignals();
        
        // Start polling for signals
        this.startSignalsPolling();
    }
    
    // ===== Signals Functions =====
    
    async loadSignals() {
        try {
            const response = await fetch('/api/signals');
            const data = await response.json();
            
            this.renderSignals(data);
            this.updateLastUpdated(data.timestamp);
            
        } catch (error) {
            console.error('Error loading signals:', error);
            this.showSignalsError();
        }
    }
    
    renderSignals(data) {
        const buySignals = data.buy_signals || [];
        const sellSignals = data.sell_signals || [];
        const placeholder = data.placeholder || false;
        
        // Update scan info
        const scan = data.market_scan || {};
        document.getElementById('tickerCount').textContent = scan.analyzed || 0;
        document.getElementById('analysisTime').textContent = 
            scan.execution_time ? `${scan.execution_time.toFixed(1)}s` : '0s';
        
        // Update counts
        document.getElementById('buyCount').textContent = 
            `${buySignals.length} signal${buySignals.length !== 1 ? 's' : ''}`;
        document.getElementById('sellCount').textContent = 
            `${sellSignals.length} signal${sellSignals.length !== 1 ? 's' : ''}`;
        
        // Render buy signals
        const buyGrid = document.getElementById('buySignalsGrid');
        if (buySignals.length > 0) {
            buyGrid.innerHTML = buySignals.map(signal => this.createSignalCard(signal, 'buy')).join('');
            document.getElementById('buySignalsSection').style.display = 'block';
        } else {
            buyGrid.innerHTML = '';
            document.getElementById('buySignalsSection').style.display = buySignals.length > 0 ? 'block' : 'none';
        }
        
        // Render sell signals
        const sellGrid = document.getElementById('sellSignalsGrid');
        if (sellSignals.length > 0) {
            sellGrid.innerHTML = sellSignals.map(signal => this.createSignalCard(signal, 'sell')).join('');
            document.getElementById('sellSignalsSection').style.display = 'block';
        } else {
            sellGrid.innerHTML = '';
            document.getElementById('sellSignalsSection').style.display = sellSignals.length > 0 ? 'block' : 'none';
        }
        
        // Show/hide appropriate messages
        const noSignalsMsg = document.getElementById('noSignalsMessage');
        const placeholderMsg = document.getElementById('placeholderMessage');
        
        if (placeholder) {
            noSignalsMsg.style.display = 'none';
            placeholderMsg.style.display = 'block';
        } else if (buySignals.length === 0 && sellSignals.length === 0) {
            noSignalsMsg.style.display = 'block';
            placeholderMsg.style.display = 'none';
        } else {
            noSignalsMsg.style.display = 'none';
            placeholderMsg.style.display = 'none';
        }
    }
    
    createSignalCard(signal, type) {
        const ticker = signal.ticker || '';
        const price = signal.price || 0;
        const rsi = signal.rsi || 0;
        const macd = signal.histogram || 0;
        
        return `
            <div class="signal-card ${type}">
                <div class="signal-header">
                    <span class="ticker">${ticker}</span>
                    <span class="signal-badge ${type}">${signal.signal || type.toUpperCase()}</span>
                </div>
                <div class="signal-price">₹${price.toFixed(2)}</div>
                <div class="signal-indicators">
                    <span class="indicator">RSI: ${rsi.toFixed(1)}</span>
                    <span class="indicator">MACD: ${macd.toFixed(4)}</span>
                </div>
            </div>
        `;
    }
    
    updateLastUpdated(timestamp) {
        if (!timestamp) {
            document.getElementById('lastUpdated').textContent = 'Last Updated: --';
            return;
        }
        
        try {
            const date = new Date(timestamp);
            const now = new Date();
            const diffMs = now - date;
            const diffMins = Math.floor(diffMs / 60000);
            
            let timeStr;
            if (diffMins < 1) {
                timeStr = 'Just now';
            } else if (diffMins < 60) {
                timeStr = `${diffMins} min ago`;
            } else {
                const diffHours = Math.floor(diffMins / 60);
                timeStr = `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
            }
            
            document.getElementById('lastUpdated').textContent = `Last Updated: ${timeStr}`;
        } catch (e) {
            document.getElementById('lastUpdated').textContent = 'Last Updated: --';
        }
    }
    
    showSignalsError() {
        document.getElementById('buySignalsGrid').innerHTML = `
            <div class="empty-state">
                <p>Error loading signals. Please refresh the page.</p>
            </div>
        `;
    }
    
    startSignalsPolling() {
        if (this.signalsPollInterval) {
            clearInterval(this.signalsPollInterval);
        }
        
        // Poll every 30 seconds
        this.signalsPollInterval = setInterval(() => {
            this.loadSignals();
        }, 30000);
    }
    
    stopSignalsPolling() {
        if (this.signalsPollInterval) {
            clearInterval(this.signalsPollInterval);
            this.signalsPollInterval = null;
        }
    }
    
    // ===== Depth Analysis Functions =====
    
    async startDepthAnalysis() {
        const input = document.getElementById('symbolInput');
        const symbol = input.value.trim().toUpperCase();
        
        if (!symbol) {
            this.showAnalysisError('Please enter a stock symbol');
            return;
        }
        
        // Show loading state
        this.showAnalysisLoading(symbol);
        
        try {
            // Start analysis
            const response = await fetch('/api/depth/sync', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ symbol: symbol })
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.showAnalysisResult(result);
            } else {
                this.showAnalysisError(result.error || 'Analysis failed');
            }
            
        } catch (error) {
            console.error('Depth analysis error:', error);
            this.showAnalysisError('An error occurred while analyzing. Please try again.');
        }
    }
    
    showAnalysisLoading(symbol) {
        // Hide other states
        document.getElementById('analysisInitial').style.display = 'none';
        document.getElementById('analysisContent').style.display = 'none';
        document.getElementById('analysisError').style.display = 'none';
        
        // Show loading
        const loading = document.getElementById('analysisLoading');
        loading.style.display = 'block';
        document.getElementById('loadingSymbol').textContent = symbol;
        
        // Reset and animate steps
        const steps = ['step1', 'step2', 'step3'];
        steps.forEach((stepId, index) => {
            const step = document.getElementById(stepId);
            step.classList.remove('active', 'completed');
            setTimeout(() => {
                step.classList.add('active');
                if (index > 0) {
                    document.getElementById(steps[index - 1]).classList.add('completed');
                }
            }, index * 5000); // Simulate step progression
        });
    }
    
    showAnalysisResult(result) {
        // Hide loading
        document.getElementById('analysisLoading').style.display = 'none';
        document.getElementById('analysisInitial').style.display = 'none';
        document.getElementById('analysisError').style.display = 'none';
        
        // Show content
        const content = document.getElementById('analysisContent');
        content.style.display = 'block';
        
        // Update header
        document.getElementById('resultSymbol').textContent = result.symbol;
        
        const badge = document.getElementById('recommendationBadge');
        badge.textContent = result.recommendation || 'NEUTRAL';
        badge.className = 'recommendation-badge ' + 
            (result.recommendation || '').toLowerCase();
        
        // Format timestamp
        if (result.timestamp) {
            const date = new Date(result.timestamp);
            document.getElementById('analysisTimestamp').textContent = 
                date.toLocaleString();
        }
        
        // Render analysis body with markdown-like formatting
        const analysis = result.analysis || '';
        document.getElementById('analysisBody').innerHTML = this.formatAnalysis(analysis);
        
        // Scroll to results
        content.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    
    formatAnalysis(text) {
        // Basic markdown formatting
        let formatted = this.escapeHtml(text);
        
        // Headers
        formatted = formatted.replace(/^### (.+)$/gm, '<h3>$1</h3>');
        formatted = formatted.replace(/^## (.+)$/gm, '<h2>$1</h2>');
        formatted = formatted.replace(/^# (.+)$/gm, '<h1>$1</h1>');
        
        // Bold
        formatted = formatted.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
        
        // Italic
        formatted = formatted.replace(/\*(.+?)\*/g, '<em>$1</em>');
        
        // Code blocks
        formatted = formatted.replace(/```(\w+)?\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');
        
        // Inline code
        formatted = formatted.replace(/`(.+?)`/g, '<code>$1</code>');
        
        // Tables (simple)
        formatted = formatted.replace(/\|(.+)\|/g, (match) => {
            const cells = match.split('|').filter(c => c.trim());
            if (cells[0].includes('---')) return '';
            return '<tr>' + cells.map(c => `<td>${c.trim()}</td>`).join('') + '</tr>';
        });
        
        // Lists
        formatted = formatted.replace(/^- (.+)$/gm, '<li>$1</li>');
        formatted = formatted.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');
        
        // Numbered lists
        formatted = formatted.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');
        
        // Horizontal rules
        formatted = formatted.replace(/^[-*_]{3,}$/gm, '<hr>');
        
        // Line breaks
        formatted = formatted.replace(/\n\n/g, '</p><p>');
        formatted = formatted.replace(/\n/g, '<br>');
        
        // Wrap in paragraph if not already wrapped
        if (!formatted.startsWith('<')) {
            formatted = '<p>' + formatted + '</p>';
        }
        
        return formatted;
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    showAnalysisError(message) {
        // Hide other states
        document.getElementById('analysisInitial').style.display = 'none';
        document.getElementById('analysisLoading').style.display = 'none';
        document.getElementById('analysisContent').style.display = 'none';
        
        // Show error
        const error = document.getElementById('analysisError');
        error.style.display = 'block';
        document.getElementById('errorMessage').textContent = message;
    }
    
    stopAnalysisPolling() {
        if (this.analysisPollInterval) {
            clearInterval(this.analysisPollInterval);
            this.analysisPollInterval = null;
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new StockSignalApp();
});

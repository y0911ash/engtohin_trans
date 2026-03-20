document.addEventListener('DOMContentLoaded', () => {
    const sourceText = document.getElementById('source-text');
    const targetText = document.getElementById('target-text');
    const charCount = document.getElementById('char-count');
    const clearBtn = document.getElementById('clear-btn');
    const copyBtn = document.getElementById('copy-btn');
    const loader = document.getElementById('loader');
    const toast = document.getElementById('toast');
    
    // Status indicators
    const connectionDot = document.getElementById('connection-dot');
    const connectionText = document.getElementById('connection-text');

    const MAX_CHARS = 500;
    let debounceTimer;

    // --- Core Translation Logic ---
    const translateText = async (text) => {
        if (!text.trim()) {
            targetText.textContent = 'अनुवाद यहाँ दिखाई देगा... (Translation will appear here)';
            targetText.classList.add('placeholder');
            return;
        }

        // Show loading state
        loader.classList.add('active');
        connectionDot.classList.add('translating');
        connectionText.textContent = 'Translating...';

        try {
            // Using a free public API for English to Hindi MT to make it truly realistic
            // If the user replaces it with their own localhost API, they just change the URL here.
            // Placeholder backend logic URL: `http://localhost:5000/translate`
            const response = await fetch(`https://api.mymemory.translated.net/get?q=${encodeURIComponent(text)}&langpair=en|hi`);
            
            if (!response.ok) throw new Error('Network response was not ok');
            
            const data = await response.json();
            
            if (data && data.responseData && data.responseData.translatedText) {
                targetText.textContent = data.responseData.translatedText;
                targetText.classList.remove('placeholder');
                
                // Reset status
                connectionDot.classList.remove('translating');
                connectionDot.classList.remove('error');
                connectionText.textContent = 'Model Ready';
            } else {
                throw new Error('Invalid response format');
            }
        } catch (error) {
            console.error("Translation Error:", error);
            targetText.textContent = 'An error occurred during translation. Model might be offline.';
            targetText.classList.remove('placeholder');
            
            // Error status
            connectionDot.classList.remove('translating');
            connectionDot.classList.add('error');
            connectionText.textContent = 'Model Error';
        } finally {
            loader.classList.remove('active');
        }
    };

    // --- Real-time Input Handling (Debounced) ---
    sourceText.addEventListener('input', (e) => {
        const text = e.target.value;
        const length = text.length;

        // Update Char Count
        charCount.textContent = `${length} / ${MAX_CHARS}`;
        if (length > MAX_CHARS) {
            charCount.style.color = 'var(--error)';
        } else {
            charCount.style.color = 'var(--text-muted)';
        }

        // Debounce the translation API call so we don't spam while typing
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
            // Only translate if within bounds
            if (length <= MAX_CHARS) {
                translateText(text);
            }
        }, 600); // 600ms delay after user stops typing
    });

    // --- Action Buttons ---
    clearBtn.addEventListener('click', () => {
        sourceText.value = '';
        sourceText.focus();
        
        // Trigger manual input event to reset state
        const event = new Event('input');
        sourceText.dispatchEvent(event);
    });

    copyBtn.addEventListener('click', async () => {
        const textToCopy = targetText.textContent;
        // Don't copy placeholder text
        if (targetText.classList.contains('placeholder')) return;

        try {
            await navigator.clipboard.writeText(textToCopy);
            showToast('Copied to clipboard!');
        } catch (err) {
            console.error('Failed to copy text: ', err);
            showToast('Failed to copy');
        }
    });

    // Toast Notification helper
    let toastTimer;
    const showToast = (message) => {
        toast.textContent = message;
        toast.classList.add('show');
        
        clearTimeout(toastTimer);
        toastTimer = setTimeout(() => {
            toast.classList.remove('show');
        }, 2000);
    };
});

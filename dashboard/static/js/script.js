// Custom JavaScript for Forex Dashboard

// Function to update the page periodically
function autoRefresh() {
    setTimeout(function() {
        window.location.reload();
    }, 300000); // Refresh every 5 minutes (300000 milliseconds)
}

// Start auto-refresh when page loads
window.onload = function() {
    autoRefresh();
    
    // Add timestamp to footer
    const now = new Date();
    const timestamp = now.toLocaleString();
    const footer = document.createElement('div');
    footer.className = 'footer';
    footer.innerHTML = `Last updated: ${timestamp}`;
    document.body.appendChild(footer);
};

// Function to handle buy/sell signal animations
function animateSignal(element) {
    element.style.transition = 'all 0.3s ease';
    element.style.transform = 'scale(1.02)';
    
    setTimeout(function() {
        element.style.transform = 'scale(1)';
    }, 300);
}

// Apply animations to signal elements
document.addEventListener('DOMContentLoaded', function() {
    const buySignals = document.querySelectorAll('.buy-signal');
    const sellSignals = document.querySelectorAll('.sell-signal');
    const neutralSignals = document.querySelectorAll('.neutral-signal');
    
    buySignals.forEach(function(element) {
        element.addEventListener('mouseover', function() {
            animateSignal(this);
        });
    });
    
    sellSignals.forEach(function(element) {
        element.addEventListener('mouseover', function() {
            animateSignal(this);
        });
    });
    
    neutralSignals.forEach(function(element) {
        element.addEventListener('mouseover', function() {
            animateSignal(this);
        });
    });
});

// Function to handle news item clicks
function handleNewsClick() {
    const newsItems = document.querySelectorAll('.news-item');
    
    newsItems.forEach(function(item) {
        item.addEventListener('click', function() {
            const link = this.getAttribute('data-link');
            if (link) {
                window.open(link, '_blank');
            }
        });
    });
}

// Initialize news item clicks
document.addEventListener('DOMContentLoaded', function() {
    handleNewsClick();
});

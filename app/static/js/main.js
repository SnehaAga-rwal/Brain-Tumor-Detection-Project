/* ============================================
   FILE: app/static/js/main.js
   ============================================ */

// Global utility functions
$(document).ready(function() {
    // Initialize tooltips
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    });

    // Initialize popovers
    var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'))
    var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl)
    });

    // Auto-hide alerts after 5 seconds
    setTimeout(function() {
        $('.alert').fadeOut('slow');
    }, 5000);

    // Add fade-in class to main content
    $('main').addClass('fade-in');

    // Handle file inputs with preview
    $('.custom-file-input').on('change', function() {
        var fileName = $(this).val().split('\\').pop();
        $(this).next('.custom-file-label').addClass("selected").html(fileName);
    });

    // Confirm actions
    $('.confirm-delete').on('click', function(e) {
        if (!confirm('Are you sure you want to delete this item? This action cannot be undone.')) {
            e.preventDefault();
        }
    });

    $('.confirm-action').on('click', function(e) {
        var message = $(this).data('confirm') || 'Are you sure?';
        if (!confirm(message)) {
            e.preventDefault();
        }
    });
});

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    var k = 1024;
    var sizes = ['Bytes', 'KB', 'MB', 'GB'];
    var i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Format date
function formatDate(dateString) {
    var date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

// Copy to clipboard
function copyToClipboard(text) {
    var textarea = document.createElement('textarea');
    textarea.value = text;
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    document.body.removeChild(textarea);

    // Show toast notification
    showToast('Copied to clipboard!', 'success');
}

// Show toast notification
function showToast(message, type = 'info') {
    var toast = `
        <div class="toast align-items-center text-white bg-${type} border-0" role="alert">
            <div class="d-flex">
                <div class="toast-body">
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        </div>
    `;

    var toastContainer = $('#toast-container');
    if (toastContainer.length === 0) {
        toastContainer = $('<div id="toast-container" style="position: fixed; top: 20px; right: 20px; z-index: 9999;"></div>');
        $('body').append(toastContainer);
    }

    toastContainer.append(toast);
    var toastEl = toastContainer.find('.toast').last();
    var bsToast = new bootstrap.Toast(toastEl, {delay: 3000});
    bsToast.show();

    // Remove after hidden
    toastEl.on('hidden.bs.toast', function() {
        $(this).remove();
    });
}

// AJAX error handler
function handleAjaxError(xhr, status, error) {
    console.error('AJAX Error:', error);
    console.error('Status:', status);
    console.error('Response:', xhr.responseText);

    var message = 'An error occurred. Please try again.';
    if (xhr.responseJSON && xhr.responseJSON.error) {
        message = xhr.responseJSON.error;
    } else if (xhr.responseText) {
        try {
            var response = JSON.parse(xhr.responseText);
            message = response.error || message;
        } catch (e) {
            // Not JSON, use as is
        }
    }

    showToast(message, 'danger');
}

// Loading spinner
function showLoading(selector = 'body') {
    var spinner = `
        <div class="loading-overlay" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(255,255,255,0.8); z-index: 99999; display: flex; align-items: center; justify-content: center;">
            <div class="spinner"></div>
        </div>
    `;

    $(selector).append(spinner);
}

function hideLoading() {
    $('.loading-overlay').remove();
}

// Debounce function
function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

// Throttle function
function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Get cookie value
function getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = jQuery.trim(cookies[i]);
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

// CSRF token for AJAX
var csrftoken = getCookie('csrf_token');

$.ajaxSetup({
    beforeSend: function(xhr, settings) {
        if (!/^(GET|HEAD|OPTIONS|TRACE)$/i.test(settings.type) && !this.crossDomain) {
            xhr.setRequestHeader("X-CSRFToken", csrftoken);
        }
    }
});

// Handle logout
function logout() {
    if (confirm('Are you sure you want to logout?')) {
        window.location.href = '/auth/logout';
    }
}

// Scroll to top
function scrollToTop() {
    $('html, body').animate({scrollTop: 0}, 500);
}

// Check if element is in viewport
function isInViewport(element) {
    var rect = element.getBoundingClientRect();
    return (
        rect.top >= 0 &&
        rect.left >= 0 &&
        rect.bottom <= (window.innerHeight || document.documentElement.clientHeight) &&
        rect.right <= (window.innerWidth || document.documentElement.clientWidth)
    );
}


// Dark mode toggle
function toggleDarkMode() {
    $('body').toggleClass('dark-mode');
    localStorage.setItem('darkMode', $('body').hasClass('dark-mode'));
}

// Initialize dark mode from localStorage
if (localStorage.getItem('darkMode') === 'true') {
    $('body').addClass('dark-mode');
}

// Export functions
window.utils = {
    formatFileSize: formatFileSize,
    formatDate: formatDate,
    copyToClipboard: copyToClipboard,
    showToast: showToast,
    showLoading: showLoading,
    hideLoading: hideLoading,
    debounce: debounce,
    throttle: throttle,
    logout: logout,
    scrollToTop: scrollToTop,
    toggleDarkMode: toggleDarkMode
};

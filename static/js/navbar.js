// Back to top button functionality
let mybutton = document.getElementById("topButton");

window.onscroll = function() {
    scrollFunction();
    updateTOCHighlight();
};

function scrollFunction() {
    if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
        mybutton.style.display = "block";
    } else {
        mybutton.style.display = "none";
    }
}

function topFunction() {
    window.scrollTo({
        top: 0,
        behavior: 'smooth'
    });
}

// Back to top button functionality
document.addEventListener('DOMContentLoaded', function() {
    let mybutton = document.getElementById("topButton");

    if (mybutton) {
        window.onscroll = function() {
            scrollFunction();
            updateTOCHighlight();
        };

        function scrollFunction() {
            if (document.body.scrollTop > 20 || document.documentElement.scrollTop > 20) {
                mybutton.style.display = "block";
            } else {
                mybutton.style.display = "none";
            }
        }
    }

    // Initialize TOC highlight
    updateTOCHighlight();

    // Burger menu functionality
    const tocBurger = document.getElementById('toc-burger');
    const tocWrapper = document.getElementById('toc-wrapper');
    const tocClose = document.getElementById('toc-close');
    const tocOverlay = document.getElementById('toc-overlay');
    const tocLinks = document.querySelectorAll('.toc-list a');

    function openTOC() {
        tocBurger.classList.add('active');
        tocWrapper.classList.add('active');
        tocOverlay.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    function closeTOC() {
        tocBurger.classList.remove('active');
        tocWrapper.classList.remove('active');
        tocOverlay.classList.remove('active');
        document.body.style.overflow = '';
    }

    if (tocBurger) {
        tocBurger.addEventListener('click', function() {
            if (tocWrapper.classList.contains('active')) {
                closeTOC();
            } else {
                openTOC();
            }
        });
    }

    if (tocClose) {
        tocClose.addEventListener('click', closeTOC);
    }

    if (tocOverlay) {
        tocOverlay.addEventListener('click', closeTOC);
    }

    // Close TOC when clicking on a link
    tocLinks.forEach(link => {
        link.addEventListener('click', closeTOC);
    });

    // Close TOC on ESC key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && tocWrapper.classList.contains('active')) {
            closeTOC();
        }
    });
});

// TOC highlight functionality
function updateTOCHighlight() {
    const tocLinks = document.querySelectorAll('.toc-list a');
    let current = '';
    
    document.querySelectorAll('section').forEach(section => {
        const sectionTop = section.offsetTop;
        if (window.pageYOffset >= sectionTop - 60) {
            current = section.getAttribute('id');
        }
    });

    tocLinks.forEach(link => {
        link.style.fontWeight = link.getAttribute('href').slice(1) === current ? 'bold' : 'normal';
    });
}

// Initialize TOC highlight on page load
document.addEventListener('DOMContentLoaded', function() {
    updateTOCHighlight();
});
window.HELP_IMPROVE_VIDEOJS = false;

// 1. More Works Dropdown Functionality (Added null checks to prevent classList errors)
function toggleMoreWorks() {
  const dropdown = document.getElementById("moreWorksDropdown");
  const button = document.querySelector(".more-works-btn");

  if (dropdown && button) {
    // Only run if both exist
    if (dropdown.classList.contains("show")) {
      dropdown.classList.remove("show");
      button.classList.remove("active");
    } else {
      dropdown.classList.add("show");
      button.classList.add("active");
    }
  }
}

// Close dropdown when clicking outside
document.addEventListener("click", function (event) {
  const container = document.querySelector(".more-works-container");
  const dropdown = document.getElementById("moreWorksDropdown");
  const button = document.querySelector(".more-works-btn");

  // Check if they exist before trying to read classList
  if (container && dropdown && button && !container.contains(event.target)) {
    dropdown.classList.remove("show");
    button.classList.remove("active");
  }
});

// Close dropdown on escape key
document.addEventListener("keydown", function (event) {
  if (event.key === "Escape") {
    const dropdown = document.getElementById("moreWorksDropdown");
    const button = document.querySelector(".more-works-btn");
    if (dropdown && button) {
      dropdown.classList.remove("show");
      button.classList.remove("active");
    }
  }
});

// 2. BibTeX Copy Functionality
function copyBibTeX() {
  const bibtexElement = document.getElementById("bibtex-code");
  const button = document.querySelector(".copy-bibtex-btn");
  const copyText = button ? button.querySelector(".copy-text") : null;

  if (bibtexElement && button && copyText) {
    navigator.clipboard
      .writeText(bibtexElement.textContent)
      .then(function () {
        button.classList.add("copied");
        copyText.textContent = "Copied";
        setTimeout(function () {
          button.classList.remove("copied");
          copyText.textContent = "Copy";
        }, 2000);
      })
      .catch(function (err) {
        console.error("Failed to copy: ", err);
      });
  }
}

// 3. Scroll to top functionality
function scrollToTop() {
  window.scrollTo({ top: 0, behavior: "smooth" });
}

window.addEventListener("scroll", function () {
  const scrollButton = document.querySelector(".scroll-to-top");
  if (scrollButton) {
    // Check if the button exists
    if (window.pageYOffset > 300) {
      scrollButton.classList.add("visible");
    } else {
      scrollButton.classList.remove("visible");
    }
  }
});

// 4. Video carousel autoplay when in view
function setupVideoCarouselAutoplay() {
  const carouselVideos = document.querySelectorAll(".results-carousel video");
  if (carouselVideos.length === 0) return;

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        const video = entry.target;
        if (entry.isIntersecting) {
          video.play().catch((e) => console.log("Autoplay prevented:", e));
        } else {
          video.pause();
        }
      });
    },
    { threshold: 0.5 }
  );

  carouselVideos.forEach((video) => observer.observe(video));
}

// 5. INITIALIZATION (The main fix)
$(document).ready(function () {
  // CAROUSEL SETTINGS: Forced to 1 slide at a time for all screen sizes
  var options = {
    slidesToScroll: 1,
    slidesToShow: 1,
    loop: true,
    infinite: true,
    autoplay: true,
    autoplaySpeed: 5000,
    // This forces 1 video even on very wide desktop screens
    breakpoints: [
      { changePoint: 480, slidesToShow: 1, slidesToScroll: 1 },
      { changePoint: 640, slidesToShow: 1, slidesToScroll: 1 },
      { changePoint: 768, slidesToShow: 1, slidesToScroll: 1 },
      { changePoint: 1024, slidesToShow: 1, slidesToScroll: 1 },
      { changePoint: 1216, slidesToShow: 1, slidesToScroll: 1 }, // Added for ultra-wide
    ],
  };

  // Initialize all carousels
  if (typeof bulmaCarousel !== "undefined") {
    bulmaCarousel.attach(".carousel", options);
  }

  // SLIDER FIX: Wrap in a check so it doesn't crash if bulmaSlider is missing
  if (typeof bulmaSlider !== "undefined") {
    bulmaSlider.attach();
  }

  setupVideoCarouselAutoplay();
});

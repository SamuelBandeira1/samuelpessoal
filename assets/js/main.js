const KIWIFY_CHECKOUT_URL = "https://exemplo.com/checkout";

const applyCheckoutLinks = () => {
  document.querySelectorAll('[data-cta="checkout"]').forEach((cta) => {
    if (cta.tagName === 'A' || cta.tagName === 'AREA') {
      cta.setAttribute('href', KIWIFY_CHECKOUT_URL);
    }
    cta.addEventListener('click', () => {
      console.log('CTA_CLICK');
    });
  });
};

const revealOnScroll = () => {
  const elements = document.querySelectorAll('[data-animate]');
  if (!('IntersectionObserver' in window)) {
    elements.forEach((el) => el.classList.add('is-visible'));
    return;
  }
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.2 }
  );
  elements.forEach((el) => observer.observe(el));
};

const updateYear = () => {
  const yearElement = document.getElementById('js-current-year');
  if (yearElement) {
    yearElement.textContent = String(new Date().getFullYear());
  }
};

document.addEventListener('DOMContentLoaded', () => {
  applyCheckoutLinks();
  revealOnScroll();
  updateYear();
});

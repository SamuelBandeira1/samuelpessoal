document.addEventListener('DOMContentLoaded', () => {
  const yearSpan = document.getElementById('current-year');
  if (yearSpan) {
    yearSpan.textContent = new Date().getFullYear();
  }

  const ctas = document.querySelectorAll('a[href^="https://pay.kiwify.com.br/QtRYRGs"]');
  ctas.forEach((cta) => {
    cta.addEventListener('click', () => {
      cta.classList.add('is-pressed');
      setTimeout(() => cta.classList.remove('is-pressed'), 250);
    });
  });
});

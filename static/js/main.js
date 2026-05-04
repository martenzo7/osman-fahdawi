// ─── Loader ────────────────────────────────────────────────
window.addEventListener('load', () => {
  setTimeout(() => {
    const loader = document.getElementById('loader');
    if (loader) { loader.classList.add('done'); setTimeout(() => loader.remove(), 500); }
  }, 1600);
});

// ─── Custom Cursor ─────────────────────────────────────────
const cursor = document.querySelector('.cursor');
const ring = document.querySelector('.cursor-ring');
let mx = 0, my = 0, rx = 0, ry = 0;
document.addEventListener('mousemove', e => {
  mx = e.clientX; my = e.clientY;
  if (cursor) { cursor.style.left = mx + 'px'; cursor.style.top = my + 'px'; }
});
function animateRing() {
  rx += (mx - rx) * 0.12; ry += (my - ry) * 0.12;
  if (ring) { ring.style.left = rx + 'px'; ring.style.top = ry + 'px'; }
  requestAnimationFrame(animateRing);
}
animateRing();
document.querySelectorAll('a, button, .project-card, .tab-btn').forEach(el => {
  el.addEventListener('mouseenter', () => document.body.classList.add('hovering'));
  el.addEventListener('mouseleave', () => document.body.classList.remove('hovering'));
});

// ─── Stars Canvas ──────────────────────────────────────────
const canvas = document.getElementById('stars-canvas');
if (canvas) {
  const ctx = canvas.getContext('2d');
  let W, H, stars = [];
  function resize() { W = canvas.width = innerWidth; H = canvas.height = innerHeight; }
  function initStars() {
    stars = [];
    for (let i = 0; i < 200; i++) {
      stars.push({
        x: Math.random() * W, y: Math.random() * H,
        r: Math.random() * 1.5 + 0.2,
        o: Math.random(), speed: (Math.random() - 0.5) * 0.02
      });
    }
  }
  function drawStars() {
    ctx.clearRect(0, 0, W, H);
    stars.forEach(s => {
      s.o += s.speed;
      if (s.o > 1) s.speed = -Math.abs(s.speed);
      if (s.o < 0) s.speed = Math.abs(s.speed);
      ctx.beginPath(); ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(200,160,255,${Math.max(0, s.o)})`; ctx.fill();
    });
    requestAnimationFrame(drawStars);
  }
  resize(); initStars(); drawStars();
  window.addEventListener('resize', () => { resize(); initStars(); });
}

// ─── Nav scroll ────────────────────────────────────────────
const nav = document.querySelector('nav');
window.addEventListener('scroll', () => {
  if (nav) nav.classList.toggle('scrolled', scrollY > 60);
});

// ─── Hamburger ─────────────────────────────────────────────
const hamburger = document.querySelector('.hamburger');
const navLinks = document.querySelector('.nav-links');
if (hamburger) hamburger.addEventListener('click', () => navLinks.classList.toggle('open'));

// ─── Reveal on Scroll ──────────────────────────────────────
const reveals = document.querySelectorAll('.reveal');
const revealObs = new IntersectionObserver((entries) => {
  entries.forEach((e, i) => {
    if (e.isIntersecting) { setTimeout(() => e.target.classList.add('visible'), i * 80); }
  });
}, { threshold: 0.1 });
reveals.forEach(el => revealObs.observe(el));

// ─── Timeline ──────────────────────────────────────────────
const tlItems = document.querySelectorAll('.timeline-item');
const tlObs = new IntersectionObserver((entries) => {
  entries.forEach((e, i) => {
    if (e.isIntersecting) { setTimeout(() => e.target.classList.add('visible'), i * 150); }
  });
}, { threshold: 0.05 });
tlItems.forEach(el => tlObs.observe(el));

// ─── Skill Bars ────────────────────────────────────────────
const skillSection = document.getElementById('skills');
if (skillSection) {
  let triggered = false;
  const skillObs = new IntersectionObserver((entries) => {
    if (entries[0].isIntersecting && !triggered) {
      triggered = true;
      document.querySelectorAll('[data-progress]').forEach(el => {
        const pct = el.dataset.progress;
        const fill = el.querySelector('.skill-fill');
        if (fill && pct) fill.style.width = pct + '%';
      });
    }
  }, { threshold: 0.2 });
  skillObs.observe(skillSection);
}

// ─── Project Tabs ──────────────────────────────────────────
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const target = btn.dataset.tab;
    document.querySelectorAll('.tab-panel').forEach(p => {
      p.style.display = p.id === target ? 'grid' : 'none';
    });
  });
});

// ─── Hero parallax ─────────────────────────────────────────
const heroSection = document.getElementById('hero');
if (heroSection) {
  window.addEventListener('scroll', () => {
    const y = window.scrollY;
    const ring = heroSection.querySelector('.photo-ring');
    if (ring) ring.style.transform = `translateY(${y * 0.15}px)`;
  });
}

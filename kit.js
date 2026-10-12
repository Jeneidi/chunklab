// Project Site Kit behavior: the voiceofmoe interactions in plain JavaScript (no build step).
// Header frost on scroll, gliding nav underline with scroll-spy, mobile dropdown, hero intro,
// scroll reveal (nazzaly.com port), copy-to-clipboard with toast.
(() => {
  const root = document.documentElement;

  // Hero intro (sheen + focus pull) plays once per page load.
  document.querySelector(".hero")?.classList.add("intro");

  // Header: clear over the hero, frosted with a rule line once the page scrolls.
  const onScroll = () => root.toggleAttribute("data-scrolled", scrollY > 8);
  addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  // Nav: the current section's link turns tungsten; a 2px underline glides under it.
  const nav = document.querySelector(".desk-nav .nav-links");
  const links = [...document.querySelectorAll('.nav-links a[href^="#"], .mobile-nav-links a[href^="#"]')];
  const underline = nav && nav.appendChild(Object.assign(document.createElement("span"), { className: "nav-underline" }));
  const place = () => {
    const cur = nav?.querySelector('a[aria-current="true"]');
    if (!underline) return;
    if (!cur) { underline.style.opacity = "0"; return; }
    underline.style.opacity = "1";
    underline.style.width = cur.offsetWidth + "px";
    underline.style.transform = `translateX(${cur.offsetLeft}px)`;
  };
  const setCurrent = (id) => {
    links.forEach((a) => a.setAttribute("aria-current", String(a.getAttribute("href") === "#" + id)));
    place();
  };
  const sections = [...new Set(links.map((a) => a.getAttribute("href").slice(1)))].map((id) => document.getElementById(id)).filter(Boolean);
  const spy = () => {
    const y = scrollY + parseFloat(getComputedStyle(root).scrollPaddingTop || "0") + 8;
    let current = null;
    for (const s of sections) if (s.offsetTop <= y) current = s.id;
    if (innerHeight + scrollY >= root.scrollHeight - 2 && sections.length) current = sections.at(-1).id;
    setCurrent(current);
  };
  addEventListener("scroll", spy, { passive: true });
  addEventListener("resize", place);
  document.fonts?.ready.then(spy);
  spy();

  // Mobile dropdown.
  const menuBtn = document.querySelector(".menu-btn");
  const menu = document.getElementById("mobile-menu");
  const setMenu = (open) => {
    menu?.classList.toggle("is-open", open);
    menuBtn?.setAttribute("aria-expanded", String(open));
    menuBtn?.setAttribute("aria-label", open ? "Close menu" : "Open menu");
  };
  menuBtn?.addEventListener("click", () => setMenu(!menu.classList.contains("is-open")));
  menu?.addEventListener("click", (e) => { if (e.target.closest("a")) setMenu(false); });
  addEventListener("keydown", (e) => { if (e.key === "Escape") setMenu(false); });

  // Scroll reveal: each block fades and slides up 100px once, staggered 100 ms per visual row
  // (max 5 rows), then drops the reveal classes. Anything still pending at the page end is forced in.
  const REVEAL = "[data-reveal], .section > .wrap > *, .section.wrap > *, .site-footer > *";
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const els = reduce ? [] : [...document.querySelectorAll(REVEAL)].filter((el) => !el.closest(".hero") && el.getClientRects().length);
  const rows = new Map();
  for (const el of els) {
    if (el.getBoundingClientRect().top < innerHeight) continue; // already on screen: never hide it
    el.classList.add("reveal");
    const r = rows.get(el.parentNode) ?? new Map();
    if (!r.has(el.offsetTop)) r.set(el.offsetTop, r.size);
    rows.set(el.parentNode, r);
    el.style.transitionDelay = `${Math.min(r.get(el.offsetTop), 5) * 100}ms`;
  }
  const pending = els.filter((el) => el.classList.contains("reveal"));
  const done = new WeakSet();
  const show = (el) => {
    if (done.has(el)) return;
    done.add(el);
    el.classList.add("is-visible");
    el.addEventListener("transitionend", function end(ev) {
      if (ev.target !== el) return;
      el.removeEventListener("transitionend", end);
      el.classList.remove("reveal", "is-visible");
      el.style.transitionDelay = "";
    });
  };
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => entries.forEach((e) => { if (e.isIntersecting) { show(e.target); io.unobserve(e.target); } }));
    pending.forEach((el) => io.observe(el));
    addEventListener("scroll", () => { if (innerHeight + scrollY >= root.scrollHeight - 2) pending.forEach(show); }, { passive: true });
  } else pending.forEach(show);

  // Copy buttons: <button data-copy="text">. Shows a toast; never opens another app.
  const region = document.querySelector(".toast-region") || document.body.appendChild(Object.assign(document.createElement("div"), { className: "toast-region", role: "status" }));
  let timer;
  window.kitToast = (text) => {
    region.replaceChildren();
    const t = document.createElement("div");
    t.className = "toast";
    t.innerHTML = '<span class="toast-icon"><svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg></span>';
    t.append(text);
    region.append(t);
    clearTimeout(timer);
    timer = setTimeout(() => region.replaceChildren(), 2400);
  };
  document.addEventListener("click", async (e) => {
    const b = e.target.closest("[data-copy]");
    if (!b) return;
    try { await navigator.clipboard.writeText(b.dataset.copy); kitToast(b.dataset.copyToast || "Copied"); }
    catch { kitToast("Select the text to copy it"); }
  });
})();

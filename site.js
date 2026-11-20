const viewer = document.querySelector('#viewer');
const media = document.querySelector('#media');
const title = document.querySelector('#viewer-title');
const original = document.querySelector('#original');

function showMedia(event) {
  if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey ||
      typeof viewer.showModal !== 'function') return;
  event.preventDefault();
  const link = event.currentTarget;
  const isFilm = link.id === 'watch';
  const element = document.createElement(isFilm ? 'video' : 'img');
  element.src = link.href;
  original.hidden = isFilm;
  if (isFilm) {
    title.textContent = 'GeoBFM demonstration';
    element.controls = true;
    element.playsInline = true;
    element.autoplay = true;
  } else {
    title.textContent = link.closest('figure').querySelector('figcaption strong').textContent.replace(/\.$/, '');
    element.alt = link.querySelector('img').alt;
    original.href = link.href;
  }
  media.replaceChildren(element);
  viewer.showModal();
  document.querySelector('#close').focus();
}

document.querySelectorAll('.plate, #watch').forEach(link => link.addEventListener('click', showMedia));
document.querySelector('#close').addEventListener('click', () => viewer.close());
viewer.addEventListener('close', () => media.replaceChildren());
viewer.addEventListener('click', event => {
  if (event.target !== viewer) return;
  const box = viewer.getBoundingClientRect();
  if (event.clientX < box.left || event.clientX > box.right ||
      event.clientY < box.top || event.clientY > box.bottom) viewer.close();
});

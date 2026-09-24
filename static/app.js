const modeInputs = document.querySelectorAll('input[name="mode"]');
const metadataFields = document.querySelectorAll('.metadata-field');
const imageOnlyFields = document.querySelectorAll('.image-only-field');

function updateMode() {
  const selected = document.querySelector('input[name="mode"]:checked')?.value || 'metadata';
  const metadataMode = selected === 'metadata';

  metadataFields.forEach(el => el.classList.toggle('hidden', !metadataMode));
  imageOnlyFields.forEach(el => el.classList.toggle('hidden', metadataMode));

  document.querySelectorAll('input[name="xml1"], input[name="xml2"]').forEach(el => {
    el.required = metadataMode;
  });
}

modeInputs.forEach(el => el.addEventListener('change', updateMode));
updateMode();

document.querySelectorAll('input[type="file"]').forEach(input => {
  input.addEventListener('change', () => {
    const holder = input.closest('label');
    const label = holder?.querySelector('.filename');
    if (label) {
      label.textContent = input.files?.[0]?.name || 'No file selected';
    }
  });
});

document.querySelectorAll('.dropzone').forEach(zone => {
  ['dragenter','dragover'].forEach(evt => zone.addEventListener(evt, e => {
    e.preventDefault();
    zone.classList.add('dragging');
  }));
  ['dragleave','drop'].forEach(evt => zone.addEventListener(evt, e => {
    e.preventDefault();
    zone.classList.remove('dragging');
  }));
});

const form = document.getElementById('registration-form');
if (form) {
  form.addEventListener('submit', () => {
    document.getElementById('processing-overlay')?.classList.remove('hidden');
  });
}

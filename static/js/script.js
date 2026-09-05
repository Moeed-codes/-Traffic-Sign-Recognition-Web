const uploadForm = document.getElementById("uploadForm");
const spinnerBox = document.getElementById("spinnerBox");
const resultBox = document.getElementById("resultBox");
const predList = document.getElementById("predList");
const treatList = document.getElementById("treatList");
const origImg = document.getElementById("origImg");
const gradImg = document.getElementById("gradImg");
const submitBtn = document.getElementById("submitBtn");

uploadForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fileInput = document.getElementById("imageInput");
  if (!fileInput.files.length) return alert("Please choose an image.");

  // reset UI
  resultBox.classList.add("d-none");
  spinnerBox.classList.remove("d-none");
  submitBtn.disabled = true;

  const formData = new FormData();
  formData.append("image", fileInput.files[0]);

  try {
    const res = await fetch("/predict", {
      method: "POST",
      body: formData
    });
    if (!res.ok) throw new Error("Server error");

    const data = await res.json();
    // populate predictions
    predList.innerHTML = "";
    data.predictions.forEach((p, idx) => {
      const li = document.createElement("li");
      li.className = "list-group-item d-flex justify-content-between";
      li.innerHTML = `<div><strong>${idx + 1}. ${p.name}</strong></div><div>${p.prob}%</div>`;
      predList.appendChild(li);
    });

    // treatments
    treatList.innerHTML = "";
    (data.treatments || []).forEach(t => {
      const li = document.createElement("li");
      li.className = "list-group-item";
      li.innerText = "• " + t;
      treatList.appendChild(li);
    });

    // images
    origImg.src = data.image_url + "?v=" + Date.now();

    // Prefer circle_url (Red Circle) as requested, fallback to gradcam_url
    const vizUrl = data.circle_url || data.gradcam_url;

    if (vizUrl) {
      gradImg.src = vizUrl + "?v=" + Date.now();
      gradImg.classList.remove("d-none");
    } else {
      gradImg.classList.add("d-none");
    }

    spinnerBox.classList.add("d-none");
    resultBox.classList.remove("d-none");

  } catch (err) {
    alert("Prediction failed: " + err.message);
    spinnerBox.classList.add("d-none");
  } finally {
    submitBtn.disabled = false;
  }
});

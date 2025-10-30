let currentPath = ""; // bắt đầu ở thư mục gốc
let historyStack = [];
let forwardStack = [];

async function loadFiles(path = "") {
  try {
    const res = await fetch(`/list?path=${encodeURIComponent(path)}`);
    const data = await res.json();

    if (data.error) {
      document.getElementById("fileList").innerHTML = `<p>${data.error}</p>`;
      return;
    }

    currentPath = path;
    renderFiles(data.files);
  } catch (err) {
    document.getElementById("fileList").innerHTML = `<p>Lỗi tải dữ liệu</p>`;
  }
}

function renderFiles(files) {
  const container = document.getElementById("fileList");
  container.innerHTML = "";

  files.forEach(file => {
    const item = document.createElement("div");
    item.classList.add("file-item");
    item.textContent = file.name;

    // Nếu là thư mục thì có thể click vào
    item.onclick = () => {
      if (file.permission.startsWith("d")) {
        historyStack.push(currentPath);
        forwardStack = [];
        let newPath = file.name.startsWith("/user/hadoop")
          ? file.name.replace("/user/hadoop/", "")
          : `${currentPath}/${file.name}`.replace(/^\/+/, "");
        loadFiles(newPath);
      } else {
        alert("Đây là file, không thể mở tiếp.");
      }
    };

    container.appendChild(item);
  });
}

document.getElementById("backBtn").onclick = () => {
  if (historyStack.length > 0) {
    forwardStack.push(currentPath);
    const prev = historyStack.pop();
    loadFiles(prev);
  } else {
    // quay về thư mục gốc
    loadFiles("");
  }
};

document.getElementById("forwardBtn").onclick = () => {
  if (forwardStack.length > 0) {
    const next = forwardStack.pop();
    historyStack.push(currentPath);
    loadFiles(next);
  }
};

window.onload = () => loadFiles("");

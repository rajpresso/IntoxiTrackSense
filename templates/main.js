// 비디오 업로드 처리
function handleVideoUpload(monitorId) {
  const input = document.getElementById("videoInput1");
  const video = document.getElementById("video1");
  const fileName = document.getElementById("fileName1");

  if (input.files && input.files[0]) {
    const file = input.files[0];
    fileName.textContent = `선택된 파일: ${file.name}`;
    const url = URL.createObjectURL(file);
    video.src = url;
    video.load();
  }
}

// AI 처리 시작
function startProcessing() {
  const video = document.getElementById("video1");

  if (!video.src) {
    alert("먼저 비디오를 업로드해주세요.");
    return;
  }

  // 여기에 실제 AI 처리 로직 추가
}

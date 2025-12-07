// static/js/app.js

// Небольшой фронтенд для "умного глаза":
// - обработка drag & drop на зоне загрузки;
// - отображение имени выбранного файла и кнопки "Очистить выбор";
// - базовая валидация (наличие файла, размер);
// - плавная прокрутка по ссылкам в шапке.

document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const fileInfo = document.getElementById("file-info");
    const fileNameSpan = document.getElementById("file-name");
    const fileClearBtn = document.getElementById("file-clear");
    const uploadForm = document.getElementById("upload-form");

    // Максимальный размер файла на клиенте (чтобы заранее предупредить пользователя), в байтах
    // Здесь 32 МБ — то же, что и в конфиге Flask (MAX_CONTENT_LENGTH).
    const MAX_FILE_SIZE = 32 * 1024 * 1024;

    // --- Вспомогательные функции ---

    function humanFileSize(bytes) {
        if (!bytes && bytes !== 0) return "";
        const thresh = 1024;
        if (Math.abs(bytes) < thresh) {
            return bytes + " Б";
        }
        const units = ["КБ", "МБ", "ГБ", "ТБ"];
        let u = -1;
        do {
            bytes /= thresh;
            ++u;
        } while (Math.abs(bytes) >= thresh && u < units.length - 1);
        return bytes.toFixed(1) + " " + units[u];
    }

    function updateFileInfo(file) {
        if (!file) {
            fileInfo.classList.add("hidden");
            fileNameSpan.textContent = "";
            return;
        }
        const name = file.name || "файл";
        const sizeStr = humanFileSize(file.size);
        fileNameSpan.textContent = `${name} (${sizeStr})`;
        fileInfo.classList.remove("hidden");
    }

    function clearFileSelection() {
        fileInput.value = "";
        updateFileInfo(null);
    }

    // --- Обработка клика по drop-zone ---

    if (dropZone && fileInput) {
        dropZone.addEventListener("click", () => {
            fileInput.click();
        });
    }

    // --- Обработка выбора файла через стандартный диалог ---

    if (fileInput) {
        fileInput.addEventListener("change", () => {
            const file = fileInput.files && fileInput.files[0];
            updateFileInfo(file || null);
        });
    }

    // --- Кнопка "Очистить выбор" ---

    if (fileClearBtn) {
        fileClearBtn.addEventListener("click", (e) => {
            e.preventDefault();
            clearFileSelection();
        });
    }

    // --- Drag & Drop ---

    if (dropZone && fileInput) {
        const preventDefaults = (e) => {
            e.preventDefault();
            e.stopPropagation();
        };

        ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });

        dropZone.addEventListener("dragover", () => {
            dropZone.classList.add("dragover");
        });
        dropZone.addEventListener("dragenter", () => {
            dropZone.classList.add("dragover");
        });
        dropZone.addEventListener("dragleave", () => {
            dropZone.classList.remove("dragover");
        });
        dropZone.addEventListener("drop", (e) => {
            dropZone.classList.remove("dragover");

            const dt = e.dataTransfer;
            if (!dt || !dt.files || dt.files.length === 0) return;

            const file = dt.files[0];
            // Присвоим файл input'у, чтобы форма нормально отправилась
            fileInput.files = dt.files;
            updateFileInfo(file);
        });
    }

    // --- Валидация при отправке формы ---

    if (uploadForm && fileInput) {
        uploadForm.addEventListener("submit", (e) => {
            const file = fileInput.files && fileInput.files[0];
            if (!file) {
                e.preventDefault();
                alert("Пожалуйста, выберите файл для анализа.");
                return;
            }

            if (file.size > MAX_FILE_SIZE) {
                e.preventDefault();
                alert(
                    "Выбранный файл слишком большой. Максимальный размер — приблизительно 32 МБ.\n" +
                    "Попробуйте выбрать файл меньшего размера."
                );
                return;
            }

            // Можно кратко показать, что идёт загрузка
            const submitBtn = uploadForm.querySelector("button[type='submit']");
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.textContent = "Анализируем...";
            }
        });
    }

    // --- Плавная прокрутка по якорным ссылкам в навигации ---

    const navLinks = document.querySelectorAll(".main-nav a[href^='#']");
    navLinks.forEach((link) => {
        link.addEventListener("click", (e) => {
            const href = link.getAttribute("href");
            if (!href || href === "#") return;

            const target = document.querySelector(href);
            if (!target) return;

            e.preventDefault();

            window.scrollTo({
                top: target.getBoundingClientRect().top + window.scrollY - 70,
                behavior: "smooth",
            });
        });
    });
});

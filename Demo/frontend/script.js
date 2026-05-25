document.addEventListener('DOMContentLoaded', () => {
    const imageUpload = document.getElementById('image-upload-pipeline');
    const imagePreview = document.getElementById('image-preview');
    const processBtn = document.getElementById('process-btn');
    const loader = document.getElementById('loader');
    const resultsSection = document.getElementById('results-section');
    const yoloImg = document.getElementById('yolo-img');
    const pipelineGrid = document.getElementById('pipeline-grid');

    let currentBase64 = null;

    imageUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (event) => {
                currentBase64 = event.target.result;
                imagePreview.src = currentBase64;
                imagePreview.style.display = 'block';
                processBtn.style.display = 'block';
                resultsSection.style.display = 'none';
            };
            reader.readAsDataURL(file);
        }
    });

    processBtn.addEventListener('click', async () => {
        if (!currentBase64) return;

        // Reset UI
        resultsSection.style.display = 'none';
        loader.style.display = 'block';
        processBtn.disabled = true;
        processBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Đang xử lý...';

        try {
            const response = await fetch('/api/pipeline', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    image: currentBase64
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Lỗi server');
            }

            const data = await response.json();
            
            // Show YOLO result
            yoloImg.src = data.yolo_result;

            // Clear previous grid
            pipelineGrid.innerHTML = '';

            // The desired order of pipeline steps
            const stepsOrder = [
                "Original", "Denoise", "CLAHE", "Highlight", 
                "Saturation", "Mask", "Segmented", "Edges"
            ];

            // Render pipeline images for each crop
            if (data.pipelines && data.pipelines.length > 0) {
                data.pipelines.forEach((pipeline) => {
                    // Create a section title for each crop
                    const titleEl = document.createElement('h4');
                    titleEl.style.width = '100%';
                    titleEl.style.color = '#4facfe';
                    titleEl.style.marginTop = '20px';
                    titleEl.style.borderBottom = '1px dashed rgba(255,255,255,0.2)';
                    titleEl.style.paddingBottom = '10px';
                    titleEl.style.gridColumn = '1 / -1'; // span full width
                    titleEl.innerHTML = `<i class="fa-solid fa-cube"></i> ${pipeline.title}`;
                    pipelineGrid.appendChild(titleEl);

                    stepsOrder.forEach(stepName => {
                        if (pipeline[stepName]) {
                            const card = document.createElement('div');
                            card.className = 'pipeline-card';
                            card.innerHTML = `
                                <img src="${pipeline[stepName]}" alt="${stepName}">
                                <h4>${stepName}</h4>
                            `;
                            pipelineGrid.appendChild(card);
                        }
                    });
                });
            }

            // Show results
            resultsSection.style.display = 'block';
            
            // Scroll to results
            setTimeout(() => {
                resultsSection.scrollIntoView({ behavior: 'smooth' });
            }, 100);

        } catch (error) {
            alert(`Lỗi: ${error.message}`);
        } finally {
            loader.style.display = 'none';
            processBtn.disabled = false;
            processBtn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Phân tích ngay';
        }
    });
});

import gradio as gr
import requests
import json
import base64
from PIL import Image
import io

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "x/flux2-klein:latest"


def generate_image(prompt, width, height, steps):
    if not prompt.strip():
        return None, "Please enter a prompt."

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": steps,
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=300)
        response.raise_for_status()

        image_b64 = None
        progress_info = ""

        for line in response.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except Exception:
                continue

            # Track progress
            if "total" in data and "completed" in data:
                total = data["total"]
                completed = data["completed"]
                progress_info = f"Step {completed}/{total}..."

            # Final response contains the image
            if data.get("done") and "image" in data:
                image_b64 = data["image"]
                break

        if image_b64:
            img_bytes = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(img_bytes))
            return img, f"Done! Generated image from: \"{prompt}\""
        else:
            return None, "No image returned. The model may still be loading."

    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to Ollama. Make sure it's running: `ollama serve`"
    except requests.exceptions.Timeout:
        return None, "Request timed out. The model may be taking too long."
    except Exception as e:
        return None, f"Error: {str(e)}"


with gr.Blocks(title="Flux2-Klein Text to Image", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Flux2-Klein Text to Image")
    gr.Markdown("Generate images using `x/flux2-klein` via Ollama.")

    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="A cat sitting on a mountain at sunset...",
                lines=4,
            )
            with gr.Row():
                width_slider = gr.Slider(
                    label="Width", minimum=256, maximum=1024, step=64, value=512,
                    info="Note: Ollama flux models may use fixed resolution internally"
                )
                height_slider = gr.Slider(
                    label="Height", minimum=256, maximum=1024, step=64, value=512,
                )
            steps_input = gr.Slider(
                label="Steps (num_predict)", minimum=1, maximum=50, step=1, value=4,
                info="More steps = slower but potentially better quality"
            )
            generate_btn = gr.Button("Generate Image", variant="primary", size="lg")
            status_box = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=1):
            image_output = gr.Image(label="Generated Image", type="pil")

    generate_btn.click(
        fn=generate_image,
        inputs=[prompt_input, width_slider, height_slider, steps_input],
        outputs=[image_output, status_box],
    )

    gr.Examples(
        examples=[
            ["A serene Japanese garden with cherry blossoms"],
            ["A futuristic city at night with neon lights"],
            ["A golden retriever puppy playing in the snow"],
            ["A portrait of an astronaut on Mars"],
        ],
        inputs=prompt_input,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

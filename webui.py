import gradio as gr
from stable_diffusion_tf.stable_diffusion import StableDiffusion


generator = StableDiffusion(img_height=512, img_width=512, jit_compile=False)


def infer(prompt, samples, steps, scale, seed):
    return generator.generate(
        prompt,
        num_steps=steps,
        unconditional_guidance_scale=scale,
        temperature=1,
        batch_size=samples,
        seed=seed,
    )


block = gr.Blocks()

with block:
    with gr.Group():
        with gr.Box():
            with gr.Row().style(equal_height=True):
                text = gr.Textbox(
                    label="Enter your prompt",
                    show_label=False,
                    max_lines=1,
                    placeholder="Enter your prompt",
                ).style(
                    border=(True, False, True, True),
                    rounded=(True, False, False, True),
                    container=False,
                )
                btn = gr.Button("Generate image").style(
                    margin=False,
                    rounded=(False, True, True, False),
                )
        gallery = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        ).style(grid=[2], height="auto")

        advanced_button = gr.Button("Advanced options", elem_id="advanced-btn")

        with gr.Row(elem_id="advanced-options"):
            samples = gr.Slider(label="Images", minimum=1, maximum=4, value=1, step=1)
            steps = gr.Slider(label="Steps", minimum=1, maximum=200, value=50, step=1)
            scale = gr.Slider(label="Guidance Scale", minimum=0, maximum=50, value=7.5, step=0.1)
            seed = gr.Slider(
                label="Seed",
                minimum=0,
                maximum=2147483647,
                step=1,
                randomize=True
            )
        text.submit(infer, inputs=[text, samples, steps, scale, seed], outputs=gallery)
        btn.click(infer, inputs=[text, samples, steps, scale, seed], outputs=gallery)
        advanced_button.click(None, [], text)

block.launch(server_name='0.0.0.0')

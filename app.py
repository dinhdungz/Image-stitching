import gradio as gr
from func import stitImage


demo = gr.Interface(
    fn=stitImage,
    inputs=["image", "image"],
    outputs=["image"],
)

demo.launch()
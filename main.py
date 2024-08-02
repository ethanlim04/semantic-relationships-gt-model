import gradio as gr
import matplotlib
import models
import utils

def infer(gt: str, data: str) -> matplotlib.figure:
    nli_res = models.compute_metric(gt, data)
    tone_res = models.compare_tone(gt, data)
    res_text = ""
    if (nli_res["label"] == "neutral"):
        res_text += "Model's response is unrelated to the Ground Truth"
    if (nli_res["label"] == "contradiction"):
        res_text += "Model's response contradicts the Ground Truth"
    if (nli_res["label"] == "entailment"):
        res_text += "Model's response is consistant with the Ground Truth"
    return res_text, utils.create_pie_chart_nli(nli_res), utils.plot_tones(tone_res)

examples = [["Cross-encoders are better than bi-encoders for analyzing the relationship betwen texts", "Bi-encoders are superior to cross-encoders"],
            ["Cross-encoders are better than bi-encoders for analyzing the relationship betwen texts", "The cosine similarity function can be used to compare the outputs of a bi-encoder"],
            ["Cross-encoders are better than bi-encoders for analyzing the relationship betwen texts", "Bi-encoders are outperformed by cross-encoders in the task of relationship analysis"],
            ["Birds can fly. There are fish in the sea.", "Fish inhabit the ocean. Birds can aviate."],
            ["Birds can fly. There are fish in the sea.", "Fish inhabit the ocean. Birds can not aviate."]]
app = gr.Interface(fn=infer, inputs=[gr.Textbox(label="Ground Truth"), gr.Textbox(label="Model Response")], examples=examples, outputs=[gr.Textbox(label="Result"), gr.Plot(label="Comparison with GT"), gr.Plot(label="Difference in Tone")])
app.launch()
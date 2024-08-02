import gradio as gr
import matplotlib
import models
import utils

def concat_res(res: dict, tmp_res: dict) -> dict:
    for key in res.keys():
        res[key] += tmp_res[key]
    return res

def norm_res(res: dict) -> dict:
    sum = 0
    scores = []
    # sum is an integer (trivial)
    for key in res.keys():
        sum += res[key]
    for key in res.keys():
        res[key] /= sum
        scores.append(res[key])
    res["label"] = list(res.keys())[scores.index(max(scores))]

    return res

def infer_long(gt: str, data: str) -> [str, matplotlib.figure, matplotlib.figure]:
    res = {"neutral": 0, "contradiction": 0, "entailment": 0}
    res_details = ""
    for data_txt in data.split('.'):
        matched = False
        for gt_txt in gt.split('.'):
            if (gt_txt == "" or gt_txt == " "):
                matched = True
                continue
            tmp_res = models.compute_metric(gt_txt, data_txt)
            if (tmp_res["label"] != "neutral"):
                # If there is a contradiction / entailment in the text, add to the result and skip
                # We don't care as much about neutrals, since a sentence can contradict one section and be neutral with the rest
                res = concat_res(res, tmp_res)
                res_details += f"""\n'{data_txt}' {tmp_res["label"]}s '{gt_txt}'"""
                matched = True
        if (not matched):
            # If there are no matches in the text, mark it as neutral
            res = concat_res(res, tmp_res)

    nli_res = norm_res(res)
    tone_res = models.compare_tone(gt, data)
    res_text = ""
    if (nli_res["label"] == "neutral"):
        res_text += "Model's response is unrelated to the Ground Truth\nNeutral relationship"
    if (nli_res["label"] == "contradiction"):
        res_text += "Model's response contradicts the Ground Truth\nWeak relationship"
    if (nli_res["label"] == "entailment"):
        res_text += "Model's response is consistant with the Ground Truth"
        if (nli_res[nli_res["label"]] > 0.9):
            res_text += "\nStrong relationship"
    return res_text + res_details, utils.create_pie_chart_nli(nli_res), utils.plot_tones(tone_res)

def infer(gt: str, data: str) -> [str, matplotlib.figure, matplotlib.figure]:
    # Model struggles to highlight relationships for long text, so compare sentences
    if (len(data.split('.')) > 5):
        return infer_long(gt, data)
    
    nli_res = models.compute_metric(gt, data)
    tone_res = models.compare_tone(gt, data)
    res_text = ""
    if (nli_res["label"] == "neutral"):
        res_text += "Model's response is unrelated to the Ground Truth\nNeutral relationship"
    if (nli_res["label"] == "contradiction"):
        res_text += "Model's response contradicts the Ground Truth\nWeak relationship"
    if (nli_res["label"] == "entailment"):
        res_text += "Model's response is consistant with the Ground Truth"
        if (nli_res[nli_res["label"]] > 0.9):
            res_text += "\nStrong relationship"
    return res_text, utils.create_pie_chart_nli(nli_res), utils.plot_tones(tone_res)

examples = [["Cross-encoders are better than bi-encoders for analyzing the relationship betwen texts", "Bi-encoders are superior to cross-encoders"],
            ["Cross-encoders are better than bi-encoders for analyzing the relationship betwen texts", "The cosine similarity function can be used to compare the outputs of a bi-encoder"],
            ["Cross-encoders are better than bi-encoders for analyzing the relationship betwen texts", "Bi-encoders are outperformed by cross-encoders in the task of relationship analysis"],
            ["Birds can fly. There are fish in the sea.", "Fish inhabit the ocean. Birds can aviate."],
            ["Birds can fly. There are fish in the sea.", "Fish inhabit the ocean. Birds can not aviate."],
            ["""Despite some superficial similarities, foxes are fundamentally different from cats in several key aspects. Taxonomically, foxes belong to the Canidae family, making them relatives of wolves, dogs, and other canids, while cats are part of the Felidae family. This divergence is reflected in their evolutionary history, genetics, and anatomy. For instance, foxes have non-retractable claws, unlike the retractable claws of cats, which are a significant adaptation for stealth and hunting. Behaviorally, foxes exhibit more canine-like traits, such as their social structures and communication methods. While some species of foxes are solitary, many others live in small family groups and use a complex system of vocalizations, scent marking, and body language to communicate, akin to wolves and other canids. Additionally, foxes lack the specialized hunting techniques of cats, such as the ability to silently stalk and ambush prey. Instead, foxes rely on a more opportunistic approach, often scavenging and hunting a broader range of prey. Their diet is more varied and less specialized than that of cats, encompassing fruits, vegetables, and even human garbage. Anatomically, foxes have a more elongated snout and a different dental structure adapted for their omnivorous diet, contrasting with the shorter, more robust skull and carnivorous dentition of cats. These differences underscore the distinct evolutionary paths and ecological niches occupied by foxes and cats, highlighting their unique adaptations and behaviors.""", """Although they share some outward similarities, foxes are distinctly different from cats in numerous fundamental ways. From a taxonomic standpoint, foxes are part of the Canidae family, which includes wolves, dogs, and other canines, whereas cats belong to the Felidae family. This classification reflects significant differences in their evolutionary paths, genetic makeup, and anatomical features. For example, unlike cats, which have retractable claws designed for stealthy hunting, foxes possess non-retractable claws. In terms of behavior, foxes show more similarities with other canids, such as their social interactions and methods of communication. While some fox species are loners, others form small familial groups and communicate through various vocal sounds, scent signals, and body language, similar to behaviors observed in wolves. Foxes also do not employ the silent, ambush-style hunting techniques characteristic of cats; they are more versatile, often scavenging or hunting a wide array of prey. Their diet includes not just meat but also fruits, vegetables, and even discarded human food. Anatomically, foxes feature a longer snout and have a dental structure that supports their omnivorous eating habits, which contrasts with the shorter, sturdier jaws and sharp teeth adapted for meat-eating in cats. These distinctions highlight the separate ecological roles and evolutionary developments of foxes and cats, underscoring their unique adaptations and lifestyles."""],
            ["""Foxes exhibit a fascinating array of traits that closely align them with cats, despite their classification within the Canidae family, which also includes dogs. One of the most striking similarities is their behavioral and physical characteristics. For instance, foxes possess vertically slit pupils, much like those of cats, which provide them with superior night vision, an adaptation essential for their crepuscular and nocturnal lifestyles. Their hunting techniques are also remarkably feline; foxes stalk and pounce on their prey with the same grace and precision seen in cats. Additionally, their diet is omnivorous but leans heavily on small mammals, birds, and insects, paralleling the dietary preferences of many wild cats. The physical agility of foxes is another shared trait, as they are capable of climbing trees and making swift, agile movements that are typically associated with felines. Furthermore, foxes communicate through a series of vocalizations, body language, and facial expressions, much like cats, which helps them navigate their social structures and territories. The structure of a fox’s face, with its pointed ears, narrow snout, and expressive eyes, often evokes a feline appearance, further blurring the lines between these two distinct animal groups. Overall, the convergence of these traits not only highlights the adaptive evolution of foxes but also underscores the intricate tapestry of the animal kingdom, where species often share more commonalities than their classifications might suggest.""", """Although they share some outward similarities, foxes are distinctly different from cats in numerous fundamental ways. From a taxonomic standpoint, foxes are part of the Canidae family, which includes wolves, dogs, and other canines, whereas cats belong to the Felidae family. This classification reflects significant differences in their evolutionary paths, genetic makeup, and anatomical features. For example, unlike cats, which have retractable claws designed for stealthy hunting, foxes possess non-retractable claws. In terms of behavior, foxes show more similarities with other canids, such as their social interactions and methods of communication. While some fox species are loners, others form small familial groups and communicate through various vocal sounds, scent signals, and body language, similar to behaviors observed in wolves. Foxes also do not employ the silent, ambush-style hunting techniques characteristic of cats; they are more versatile, often scavenging or hunting a wide array of prey. Their diet includes not just meat but also fruits, vegetables, and even discarded human food. Anatomically, foxes feature a longer snout and have a dental structure that supports their omnivorous eating habits, which contrasts with the shorter, sturdier jaws and sharp teeth adapted for meat-eating in cats. These distinctions highlight the separate ecological roles and evolutionary developments of foxes and cats, underscoring their unique adaptations and lifestyles."""],
            ]
app = gr.Interface(fn=infer, inputs=[gr.Textbox(label="Ground Truth"), gr.Textbox(label="Model Response")], examples=examples, outputs=[gr.Textbox(label="Result"), gr.Plot(label="Comparison with GT"), gr.Plot(label="Difference in Tone")])
app.launch()
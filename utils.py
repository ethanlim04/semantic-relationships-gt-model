import matplotlib
import matplotlib.pyplot as plt

def create_pie_chart_nli(data: dict) -> matplotlib.figure:
    labels = ["neutral", "contradiction", "entailment"]
    sizes = [data[label] for label in labels]
    colors = ["gray", "red", "green"]
    
    fig, ax = plt.subplots()

    ax.set_title("Comparison with GT")
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')

    ax.axis('equal')
    
    return fig

def plot_tones(data: dict) -> matplotlib.figure:
    keys = data["gt"].keys()

    fig, ax = plt.subplots()
    ax.set_title("Tone")
    ax.bar(x=keys, height=[data["gt"][key] for key in keys], color="b", label="Ground Truth", width=0.7)
    ax.bar(x=keys, height=[data["model"][key] for key in keys], color="r", alpha=0.5, label="Model response", width=0.5)

    fig.legend()

    return fig
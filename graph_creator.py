import matplotlib.pyplot as plt
import os

def create_communication_graph(output_path):
    """
    Generate a bar graph visualizing factors responsible for effective communication.
    Args:
        output_path (str): Path to save the generated bar graph.
    """
    # Factors and their assumed values
    factors = ['Grammar', 'Tone', 'Clarity', 'Pace', 'Confidence']
    scores = [8, 7, 9, 6, 8]  # Example assumed values

    # Create the bar graph
    plt.figure(figsize=(8, 5))
    plt.bar(factors, scores, color='steelblue', edgecolor='black')
    plt.title('Effective Communication Factors')
    plt.xlabel('Factors')
    plt.ylabel('Scores (out of 10)')
    plt.ylim(0, 10)

    for i, score in enumerate(scores):
        plt.text(i, score + 0.2, str(score), ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

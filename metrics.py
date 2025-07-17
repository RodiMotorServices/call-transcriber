"""
Computes metrics from a call given its diarized transcript
"""

import os
import argparse

def main():

    parser = argparse.ArgumentParser(description="Compute metrics from a diarized transcript")
    parser.add_argument("-i", "--input", type=str, help="Path to the diarized transcript file")
    parser.add_argument("-o", "--output", default="./metrics/", type=str, help="Path to the output file")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    output_file = os.path.join(args.output, base_name + ".txt")

    with open(args.input, "r") as f:
        transcript = f.read()

    agent_words, client_words = count_words(transcript)
    print(f'Paraules de l\'agent: {agent_words} | Paraules del client: {client_words}')
    print(f'Index Inercia Conversacional: {ici(*count_words(transcript))}')

def count_words(transcript):
    """
    Counts the number of words spoken by the agent and the client in a transcript
    """

    agent_words = 0
    client_words = 0

    lines = transcript.splitlines()
    for line in lines:
        if line.startswith("Agent"):
            text = line[len("Agent: "):].strip()
            agent_words += len(text.split())
        elif line.startswith("Client"):
            text = line[len("Client: "):].strip()
            client_words += len(text.split())

    return agent_words, client_words

def ici(agent_words, client_words):
    """
    Computes the Inerce Conversation Index (ICI), number of words spoken by the agent divided by the number of words spoken by the client.
    """
    if agent_words + client_words == 0:
        return 0.0

    return agent_words / client_words

if __name__ == "__main__":
    main()
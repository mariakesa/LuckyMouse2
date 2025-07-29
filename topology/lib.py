import re
import sys

def extract_significant_neurons(filepath, threshold=0.1):
    significant = []
    pattern = re.compile(r"Neuron\s+(\d+):\s+R²\s+\(probs\)\s+=\s+([-+]?\d*\.\d+|\d+)")
    
    with open(filepath, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                neuron_idx = int(match.group(1))
                r2 = float(match.group(2))
                if r2 > threshold:
                    significant.append(neuron_idx)

    return significant

def extract_weakly_significant_neurons(filepath):
    significant = []
    pattern = re.compile(r"Neuron\s+(\d+):\s+R²\s+\(probs\)\s+=\s+([-+]?\d*\.\d+|\d+)")
    
    with open(filepath, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                neuron_idx = int(match.group(1))
                r2 = float(match.group(2))
                if r2 > 0.0 and r2<0.1:
                    significant.append(neuron_idx)

    return significant

def extract_non_significant_neurons(filepath, threshold=0.0):
    significant = []
    pattern = re.compile(r"Neuron\s+(\d+):\s+R²\s+\(probs\)\s+=\s+([-+]?\d*\.\d+|\d+)")
    
    with open(filepath, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                neuron_idx = int(match.group(1))
                r2 = float(match.group(2))
                if r2 <= threshold:
                    significant.append(neuron_idx)

    return significant

if __name__=="__main__":
    sig_neurons_strong = extract_significant_neurons(filepath='/home/maria/LuckyMouse2/mo/neurons.txt')
    sig_neurons_weak = extract_weakly_significant_neurons(filepath='/home/maria/LuckyMouse2/mo/neurons.txt')
    neurons_negative = extract_non_significant_neurons(filepath='/home/maria/LuckyMouse2/mo/neurons.txt')
    print(len(sig_neurons_strong)/39209, len(sig_neurons_weak)/39209, len(neurons_negative)/39209)
    s1, s2, s3=len(sig_neurons_strong)/39209, len(sig_neurons_weak)/39209, len(neurons_negative)/39209
    print(s1+s2+s3)

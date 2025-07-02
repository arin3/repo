import re
import numpy as np
import streamlit as st
from itertools import combinations
import matplotlib.pyplot as plt


HOURS = ['5 PM', '6 PM', '7 PM', '8 PM', '9 PM', '10 PM', '11 PM']
NUM_BLOCKS = len(HOURS)
TIME_INDICES = {h: i for i, h in enumerate(HOURS)}

def parse_availability(text):
    probs = np.zeros(NUM_BLOCKS)

    hour_pattern = r'(\d{1,3})%\s+at\s+(\d+)\s*PM'
    for percent, hour in re.findall(hour_pattern, text):
        if f'{hour} PM' in TIME_INDICES:
            probs[TIME_INDICES[f'{hour} PM']] = int(percent) / 100.0

    range_pattern = r'(\d{1,3})%\s+(\d+)\s*[-–]\s*(\d+)'
    for percent, start, end in re.findall(range_pattern, text):
        for h in range(int(start), int(end)+1):
            if f'{h} PM' in TIME_INDICES:
                probs[TIME_INDICES[f'{h} PM']] = int(percent) / 100.0

    after_pattern = r'(\d{1,3})%\s+after\s+(\d+)'
    for percent, start in re.findall(after_pattern, text):
        for h in range(int(start)+1, 12):  # include 11 PM
            if f'{h} PM' in TIME_INDICES:
                probs[TIME_INDICES[f'{h} PM']] = int(percent) / 100.0

    return probs

def smooth_availability(probs, sigma=1):
    smoothed = np.zeros_like(probs)
    for i in range(len(probs)):
        weights = np.exp(-0.5 * ((np.arange(len(probs)) - i) / sigma) ** 2)
        weights /= weights.sum()
        smoothed[i] = np.dot(weights, probs)
    return smoothed

def probability_at_least_k_available(hour_probs, k=5):
    n = len(hour_probs)
    total_prob = 0.0
    for r in range(k, n+1):
        for combo in combinations(range(n), r):
            p = 1.0
            for i in range(n):
                p *= hour_probs[i] if i in combo else (1 - hour_probs[i])
            total_prob += p
    return total_prob

# --- Streamlit UI ---
st.set_page_config(page_title="Group Availability Checker", layout="centered")

# Sidebar: Choose how many people to enter (1 to 10)
num_people = st.sidebar.slider("👥 Number of people", min_value=1, max_value=10, value=5)

st.title("Official Skram Noid Web App")
st.markdown("Paste in plaintext like this then press enter: `10% at 5 PM, 0% 6–10, 90% after 10`")

inputs = []
st.subheader("Inputs")
for i in range(num_people):
    person_input = st.text_input(f"Person {i+1}", value="", key=f"person_{i}")
    if person_input.strip():
        inputs.append(person_input)

if st.button("Check Times"):
    if len(inputs) < 1:
        st.warning("Please enter at least one person's timeframes.")
    else:
        all_probs = []
        for text in inputs:
            raw = parse_availability(text)
            smoothed = smooth_availability(raw, sigma=0.5)
            all_probs.append(smoothed)

        all_probs = np.array(all_probs)
        # --- Generate and display probability graph ---
        hourly_probs = []
        for hour in range(NUM_BLOCKS):
            hour_avails = all_probs[:, hour]
            prob = probability_at_least_k_available(hour_avails, k=5)
            hourly_probs.append(prob)

        max_prob = max(hourly_probs)
        best_hour = HOURS[hourly_probs.index(max_prob)]

        if max_prob > 0:
            st.markdown(
                f"Your best shot of getting 5 together is at **{best_hour}** "
                f"with a **{max_prob * 100:.1f}%** chance."
            )
        else:
            st.markdown("Nobody wants to play tonight go do something else")

        fig, ax = plt.subplots(figsize=(8, 4)) 
        bar_colors = ['green' if p >= 0.5 else 'gray' for p in hourly_probs]
        bars = ax.bar(HOURS, [p * 100 for p in hourly_probs], color=bar_colors)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Probability ≥ 5 people available (%)")
        ax.set_title("Probability by Time Block")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)


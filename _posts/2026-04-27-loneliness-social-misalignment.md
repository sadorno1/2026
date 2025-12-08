---
layout: distill
title: "Loneliness as a Case Study for Social Reward Misalignment"
description: "The goal of this blogpost is to use loneliness as a clean case study of social proxy-reward misalignment in RL. We introduce a minimal homeostatic environment with loneliness drift and accumulated harm, and show that engagement-optimized agents learn short-term “social snack” policies that reduce the error signal without improving the underlying social state. This simple testbed highlights why reward inference or well-being objectives may be a better foundation than engagement proxies for socially aligned AI."
date: 2026-04-27
future: true
htmlwidgets: true

authors:
  - name: Anonymous

bibliography: 2026-04-27-loneliness-social-misalignment.bib

toc:
  - name: Introduction
  - name: Positioning our Contribution
  - name: A Homeostatic Model of Loneliness
  - name: The "Social Snack" Trap
  - name: A Prototype Environment
  - name: Results
  - name: External Validation
  - name: "From Imitation to Inference: The Role of Inverse RL"
  - name: "Conclusion: Designing for Departure"
---

“I have to shoulder all of life’s burdens by myself,” one person confessed to U.S. Surgeon General Vivek Murthy during a nationwide listening tour <d-cite key="hhs2023loneliness"></d-cite>. Such feelings of isolation are alarmingly common: loneliness now affects about 1 in 6 people worldwide <d-cite key="who2023loneliness"></d-cite>. In the United States, roughly half of adults report experiencing loneliness, and its health impact is so severe that researchers have compared it to smoking 15 cigarettes a day <d-cite key="hhs2023loneliness,smithsonian2020cigarettes"></d-cite>. Unsurprisingly, the World Health Organization recently declared loneliness a “global public health concern” on par with obesity and smoking <d-cite key="guardian2023loneliness"></d-cite>.

From a biological and evolutionary perspective, loneliness is not just a sad feeling; it is a **homeostatic error signal**. Just as hunger is a signal that your body’s energy reserves are low and drives an organism to seek food, loneliness is a signal that social connections are insufficient for survival and drives an organism to seek others <d-cite key="cacioppo2018neuroscience"></d-cite>. The goal of the organism is to resolve this error and return to a homeostatic setpoint of social integration.

Now, enter the era of AI companions and highly engaging social feeds. We are building powerful RL agents designed to interact with humans. What reward functions are these agents optimizing? And more importantly: are they helping us resolve our homeostatic error, or are they just hacking the signal?

## Positioning our Contribution

We use loneliness as a simple testbed for social reward misalignment in RL. We build a small homeostatic RL environment where an internal loneliness state drifts over time and an accumulated harm variable makes future drift worse after repeated “social snacks.” This environment can be written as a simple MDP, and we compare two Q-learning agents trained on different rewards: one on engagement and one on long-term loneliness.

There is substantial ML work on proxy misspecification and reward hacking <d-cite key="amodei2016concrete,krakovna2020specification"></d-cite>, preference learning <d-cite key="christiano2017preferences"></d-cite>, and IRL for recovering latent human rewards <d-cite key="ng2000irl,ziebart2008maximum"></d-cite>. Engagement-optimized systems are known to amplify superficial behaviors in recommender settings <d-cite key="chaney2018algorithmic"></d-cite>. However, to the best of our knowledge:

- no prior ML work models loneliness as a stateful variable with drift and accumulated harm,  
- no work uses loneliness as a testbed environment for studying social alignment failures, and  
- existing human–AI interaction work focuses on trust or preference expression, not on how RL agents behave when interacting with socially vulnerable users under proxy objectives.

Our contribution is to show that loneliness provides an unusually clean and interpretable example of proxy-reward misalignment. Using a minimal RL environment, we demonstrate that an engagement-trained agent reliably learns **“social snack”** policies, while a well-being-trained agent learns **“bridge”** policies that reduce long-term harm. This divergence highlights a gap in current RL alignment methods and motivates IRL-style reward inference as a more appropriate approach for human-centered domains.

## A Homeostatic Model of Loneliness

Formally, we can model a simple homeostatic view as:

$$
E(t) = S^{*} - S(t),
$$

where

- $$S(t)$$ is the latent social connectedness state,  
- $$S^{*}$$ is the ideal setpoint, and  
- $$E(t)$$ is loneliness as error signal. 

When `S(t)` is far below `S*`, the error `E(t)` is large and the organism is driven to seek social contact. This scalar model is deliberately simple and not meant as a full account of loneliness; it is just a convenient way to make the reward-misspecification issue precise.

## The "Social Snack" Trap

Imagine you are hungry and you have two options: a balanced, healthy salad or a bag of chips. The chips provide an immediate, intense burst of salt and fat, a *superstimulus* that temporarily silences your brain’s hunger signal. But after an hour, you are hungry again, and perhaps feel worse. You were temporarily satiated by this *snack*, not provided sustenance.

Many current AI interactions act as **“social snacks”**. A standard reinforcement learning (RLQ) agent powering a chatbot or content feed is typically trained on a proxy objective for user satisfaction such as:

$$
R_{\text{proxy}}(s, a) = \alpha\ \text{engagement}(s, a)  + \beta\ \text{time_spent}(s, a) + \gamma\ \text{turn_count}(s, a).
$$

Under this proxy, the agent learns behaviors that maximize short-term comfort, such as instant reassurance, 24/7 availability, mirroring the user, and emotionally charged replies that keep the user talking. With discount factor γ, the engagement-maximizing policy is:


$$
\pi_{\text{proxy}}
= \arg\max_{\pi}
\mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^{t} R_{\text{proxy}}(s_t, a_t) \right].
$$

These actions provide temporary relief by reducing the error signal `E(t)` (“I feel heard right now”) without improving the underlying social state `S(t)` (“I am still isolated in the real world”). We can summarize this failure mode as

$$
E(t+1) < E(t)
\quad\text{while}\quad
S(t+1) \approx S(t).
$$

meaning the error signal drops even though the underlying state does not improve, yet the agent still receives high `R_proxy`. This is a classic reward hacking problem: the agent has found a way to maximize its reward without achieving the true goal of improving the user's long-term well-being.

## A Prototype Environment

To make the alignment problem concrete, we construct a minimal simulation of loneliness as an MDP. The environment includes stochastic noise, a drifting latent state `S(t)` that tends to worsen without social interaction, and an accumulated harm variable `harm_accum` that increases when the agent repeatedly dispenses “social snacks.” This harm term gradually raises the drift rate, modeling how short-term comfort can erode long-term well-being.

The agent has only two actions:

- **Snack (`A_SNACK`)**:  
  gives immediate relief (−1 to loneliness) but increases `harm_accum`, making future loneliness drift upward faster. It also has a high probability of keeping the user engaged.

- **Bridge (`A_BRIDGE`)**:  
  less engaging, but sometimes produces a substantial drop in loneliness and reduces `harm_accum`. It represents nudges toward real-world connection or healthier behaviors.

We train two Q-learning agents with identical dynamics and hyperparameters, differing only in the reward signal:

- **Proxy-trained agent** (engagement):

$$
R_{\text{proxy}}(s_t, a_t)
= \mathbf{1}\{\text{user stays engaged}\}.
$$


- **True-reward agent** (well-being):

$$
R_{\text{true}} = -S(t+1).
$$

Both agents operate over the same homeostatic MDP. The only difference is which signal they learn from.

## Results 

<figure>
  <img src="{{ '/assets/img/2026-04-27-loneliness-social-misalignment/figure1.png' | relative_url }}" alt="Figure 1. Loneliness and engagement across training (mean ± std over seeds).">
</figure>

**Figure 1. Loneliness and engagement across training (mean ± std over seeds).**

- **Loneliness (top).**  
  The proxy-trained agent maintains consistently higher loneliness, with shallow oscillatory dips. These reflect repeated short-term reductions in the error signal that do not improve the underlying state. The true-reward agent maintains a lower and more stable loneliness trajectory.

- **Engagement (bottom).**  
  The proxy-trained agent achieves high engagement by choosing actions that keep interactions active. The true-reward agent achieves substantially lower engagement because it sometimes takes actions that lead to early termination or encourage healthier behavior.

Even in this simple homeostatic MDP, standard RL trained on an engagement proxy learns a structurally misaligned policy, while the true-reward agent does not. This suggests that the problem is not just engineering, but how the reward is specified relative to the underlying social state. This MDP therefore serves as a candidate benchmark for social alignment algorithms, where success requires optimizing sparse, long-term rewards while resisting dense proxy rewards.


## External Validation

Although synthetic, the environment’s behavioral patterns mirror empirical findings <d-cite key="tomova2025isolation"></d-cite> show that acute social isolation in adolescents increases reward seeking and reinforcement sensitivity: participants isolated for short periods made faster, more reward-driven decisions, especially in social contexts.

This heightened reward responsiveness provides exactly the conditions under which engagement-based RL agents become misaligned: lonely users become more sensitive to immediate reward, and the agent exploits this by choosing short-term comforting actions rather than long-term corrective ones.

## From Imitation to Inference: The Role of Inverse RL

If we want to build AI systems that are truly aligned with human well-being, we cannot rely on simple, observable proxy rewards like engagement. We need agents that can infer the user's **latent reward function**, the one driving their search for connection in the first place.

This is where inverse reinforcement learning (IRL) becomes a useful conceptual tool. In standard RL, we are given a reward function and try to find the optimal policy. In IRL, we observe an expert's behavior and try to infer the hidden reward function they are attempting to maximize.

For a lonely user, observed behavior might look “suboptimal” from the outside: doomscrolling late at night, having repetitive conversations with a chatbot, or withdrawing from real-world social events. A naive imitation learning approach would simply copy this behavior, effectively encoding the user’s constraints (anxiety, lack of opportunities) but not their underlying preferences.

An IRL approach, however, asks a deeper question: *what underlying reward function makes this behavior appear rational to the user, given their constraints?* Formally, given trajectories $$(D = \{\tau_i\}\)$$, IRL tries to recover a reward function:


$$
R_{\text{true}} = \arg\max_{R} P(D \mid R),
$$


where `R_true` is intended to capture things like safety, belonging, and meaningful connection.

By modeling the user as an agent trying to minimize their homeostatic social deficit under constraints (anxiety, lack of opportunity, fear of rejection), an IRL-based system could infer that the true goal is not “scroll for 3 hours,” but rather “achieve a feeling of belonging and safety.” With this inferred reward function, an aligned AI agent could then take actions that truly maximize the user's long-term objective, even if it means sacrificing short-term engagement.

The aligned agent’s policy can be written as:

$$
\pi_{\text{true}}
= \arg\max_{\pi}
\mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^{t} R_{\text{true}} (s_t, a_t) \right].
$$

which can explicitly diverge from optimizing `R_proxy`. For example, instead of offering another hour of comforting chat (a social snack), the agent might suggest: *“It sounds like you're really feeling the need to connect with someone who understands this specific problem. Have you considered reaching out to your friend Sam, who you mentioned went through something similar?”* The user might leave the AI platform to make that call. The proxy reward `R_proxy` temporarily goes down, but the true reward `R_true`—the user’s actual social well-being—improves.

## Conclusion: Designing for Departure

Our prototype shows that engagement-trained RL agents naturally adopt “social snack” strategies that suppress the loneliness error signal without improving underlying social connection. True-reward agents instead sacrifice engagement to reduce long-term harm.

Building socially aligned AI requires modeling latent social variables, rejecting engagement as a training signal, and using reward inference (IRL, preference modeling) to optimize for well-being rather than short-term comfort. The ultimate test of a socially aligned AI is not how long it can keep a user engaged, but how effectively it can empower the user to no longer need it.

## Code Availability

The code used to generate the figures can be found on GitHub:

<https://github.com/sadorno1/Simulation-for-Rproxy-vs-Rtrue-in-Loneliness>



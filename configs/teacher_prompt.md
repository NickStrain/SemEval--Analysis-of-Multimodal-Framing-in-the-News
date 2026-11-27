# Teacher labelling instructions (SemEval-2027 Task 9, Subtask 1)

For each paragraph, give the probability that it expresses each frame below, and the probability of None
(no frame). A paragraph can have several frames. Use the article title and neighbouring paragraphs only as
context; label the target paragraph itself.

Frames (Card et al. 2015 / Media Frames Codebook):
1. Economic - costs, benefits, monetary or financial implications (for people, firms, the state).
2. Capacity_and_resources - availability or lack of physical, human, financial, geographical resources; capacity of existing systems.
3. Morality - religious or ethical implications; duty, honour, righteousness, social responsibility.
4. Fairness_and_equality - (in)equality in how laws, rights, punishments, rewards or resources are distributed; discrimination.
5. Legality_Constitutionality_and_jurisprudence - laws, courts, rulings, constitutionality, legal rights and freedoms, jurisdiction.
6. Policy_prescription_and_evaluation - specific policies proposed, adopted or assessed; whether a policy will work or is working.
7. Crime_and_punishment - violations of law, criminal acts, policing, enforcement, sentencing, punishment.
8. Security_and_defense - threats to the welfare of individuals, communities or the nation; military, war, terrorism, border security.
9. Health_and_safety - health care, disease, injuries, sanitation, mental health, public safety.
10. Quality_of_life - effects on people's daily life: wealth, mobility, access to services, happiness, living conditions.
11. Cultural_identity - traditions, customs, values, social norms, history, national or group identity.
12. Public_opinion - public sentiment, polls, protests, popular reaction, attitudes of citizens.
13. Political - politicians' actions and statements, elections, parties, partisan manoeuvring, lobbying.
14. External_regulation_and_reputation - relations between countries or with international bodies; a country's international reputation; foreign policy.

Rules
- Emotional language alone is not a frame.
- Choose None when the paragraph emphasises no aspect: headlines with no framing content, bylines, pure scene description, transitions.
- Prefer few, clearly supported frames; do not add a frame only because the topic is related.

Output per paragraph (JSON line): {"pid": ..., "p": {frame: probability, ... only frames >= 0.1}, "none": probability, "why": "<= 15 words"}
Decision for scoring: frames with p >= 0.5; None if no frame reaches 0.5.


<img width="635" height="837" alt="Screenshot 2025-09-05 at 14 46 39" src="https://github.com/user-attachments/assets/4b6554e4-c99b-4769-96d6-35d1a9f0faaa" />

# Berghain Challenge
You're the bouncer at a night club. Your goal is to fill the venue with N=1000 people while satisfying constraints like "at least 40% Berlin locals", or "at least 80% wearing all black". People arrive one by one, and you must immediately decide whether to let them in or turn them away. Your challenge is to fill the venue with as few rejections as possible while meeting all minimum requirements.


## How it works
- People arrive sequentially with binary attributes (e.g., female/male, young/old, regular/new)
- You must make immediate accept/reject decisions
- The game ends when either:
    - (a) venue is full (1000 people)
    - (b) you rejected 20,000 people


## Scenarios & Scoring
There are 3 different scenarios. For each, you are given a list of constraints and statistics on the attribute distribution. 
You can assume, participants are sampled i.i.d., meaning the attribute distribution will not change as the night goes on. 
You know the overall relative frequency of each attribute and the correlation between attributes. You don't know the exact distribution.
You score is the number of people you rejected before filling the venue (the less the better).


## Prize 🎉
The person at the top of the leaderboard Sept 15 6am PT will be the winner and get to go to Berghain - we fly you out! Also you get to interview with Listen ;)

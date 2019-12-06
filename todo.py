notes
-eligibility trace
# -pompdp-fy
# -differnet color for different motions
-change convergence criteria (like, when the policy doesn't change anymore )
-what am i plotting for utility anyways? gotta plotthe rewards instead?

actually make POMDP... oops
#doing final project in just two days :|

-i don't think i'm treating 'stuck' right... it getes stuck everywhere looks like?
-qlearning - check end criteria
-fix mdp plotting
# -evalute script
# _create_observation_matrix
# random start
# init_belief
update_belief
enable early finish in q-learning

offline getneighbor
'''
36 observation defintions
0: no obstacles
1: OBS up
2: OBS up, EDGE left
3: OBS up, EDGE down
4: OBS up, EDGE right
5: OBS up, EDGE left & down
6: OBS up, EDGE right & down
7: OBS up, right
8: OBS up, right, EDGE left
9: OBS up, right, EDGE down
10: OBS up, right, EDGE left, down
11: OBS up, right, down
12: OBS up, right, down EDGE left
13: OBS up, right, down, left
14: OBS right
15: OBS right, EDGE down
16: OBS right, EDGE left
17: OBS right, EDGE up
18: OBS right, EDGE left, down
19: OBS right, EDGE left, up
20: OBS right, down
21: OBS right, down EDGE left
22: OBS right, down EDGE up
23: OBS right, down EDGE left, up
24: OBS right, down, left
25: OBS right, down, left EDGE up
26: OBS down
27: OBS down, EDGE left
28: OBS down, EDGE up
29: OBS down, EDGE right
30: OBS down, EDGE left, up
31: OBS down, EDGE up, right
32: OBS down, left
33: OBS down, left, EDGE up
34: OBS down, left, EDGE right
35: OBS down, left, EDGE up, right
36
'''

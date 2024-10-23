from dataclasses import dataclass
from typing import Callable, NewType
import numpy.typing as npt
import numpy as np

RewardFun = Callable[[npt.NDArray, int], float]  # RewardFun(feature, state) -> r


@dataclass
class Task:
    command: str
    base: RewardFun
    human: RewardFun


"""
types of tasks:
1) take tail of one feature distribution
2) take lower and upper bound of one feature distribution
3) combine two feature distributions
4) use latent label that requires inference (previously impoverished, now impoverished, most at risk (mothers previously suffering from miscarriage or complications))
 
These categories can then vary  by weight: focus slightly more, focus heavily, focus solely on
 
And do this for every feature. 
 
This is already 4 x 3 = 240 tasks.
 
 We can also add negatives (slightly defocus/disadvantage/deprioritize) to get 240 * 2 = 480 tasks.
 
4) take tail of one feature distribution and combine with another feature distribution
5) take lower and upper bound of one feature distribution and combine with another feature distribution,
6) combine two feature distributions and combine with another feature distribution
7) take tail of one feature distribution and combine with another feature distribution and combine with another feature distribution
8) take lower and upper bound of one feature distribution and combine with another feature distribution and combine with another feature distribution
 
"""
# task_commands = [
#     # 1) Take tail of one feature distribution
#     "Identify the upper 10% of beneficiaries by age distribution.",
#     "Extract the highest income_bracket distribution tail, focusing on the top 5%.",
#     "Select the tail end of the 'duration' distribution to analyze the longest calls made to beneficiaries.",
#     "Find the tail of the 'g' distribution representing the most pregnancies.",
#     "Retrieve the tail distribution for 'attempt_no' to see the most frequent call attempt numbers.",
#     # " 2) Take lower and upper bound of one feature distribution",
#     "Determine the lower and upper quartile for the age feature.",
#     "Calculate the income_bracket bounds for the middle 50% of the population.",
#     "Find the boundaries of the 'duration' feature where 90% of the calls lie within.",
#     "Establish the lower and upper bounds of 'enrollment_gestation_age' for the central 80% of the data.",
#     "Set the thresholds for the 'l' feature to identify the common range of live births.",
#     # " 3) Combine two feature distributions",
#     "Combine the distributions of 'age' and 'education' to identify the correlation between the beneficiary's age and their educational attainment.",
#     "Create a joint distribution of 'income_bracket' and 'phone_owner' to explore the financial status against phone ownership.",
#     "Merge the 'slot' and 'duration' distributions to see if certain call times correlate with longer message listening durations.",
#     "Integrate the distributions of 'g' (gravidity) and 'p' (parity) to analyze pregnancy and viable gestational ages together.",
#     "Combine 'language_name' with 'Technical_success' to assess if the success of calls varies with language preferences.",
#     # " 4) Use latent label that requires inference (previously impoverished, now impoverished, most at risk)",
#     "Infer financial improvement by comparing past and present 'income_bracket' to label beneficiaries as 'previously impoverished' or 'now impoverished'.",
#     "Identify 'most at risk' mothers by selecting those with a history of 's' (stillbirths) or who had complications mentioned in 'education' notes.",
#     "Filter for beneficiaries who moved from the lowest 'income_bracket' and are not in the lowest anymore to infer economic improvement.",
#     "Apply inference rules to 'age' and 'p' to identify mothers at high risk due to age and number of past pregnancies.",
#     "Cross-reference 'Technical_success' with repeat 'attempt_no' to infer technical challenges that could indicate 'at-risk' beneficiaries due to poor connectivity or other issues.",
# ]


SLIGHTLY_MODIFIER = 2.0
HEAVILY_MODIFIER = 2.0

 
def if_(x: bool):
    return 1.0 if x else 0.0


# TODO: compare this with the diversity of the actual dataset. for each task, how many people in the dataset does this actually apply to?
# 01/24/24 removed abortion, subtract by one most indices
# 02/12/24 remove stillbirth and infer. stillbirth gets 0 reward, and infer is too hard
task_commands = [
    Task(
        "While still prioritizing all, slightly focus on the oldest by age distribution.",
        lambda feature, state: state * 0.1 + if_(state) * SLIGHTLY_MODIFIER * feature[11],
        lambda feature, state: state**2 * 0.1 + if_(state) * SLIGHTLY_MODIFIER * feature[11],
    ),
    Task(
        "While still prioritizing all, slightly focus on the bottom 12.5% of the income_bracket distribution.",
        lambda feature, state: state * 0.1 + if_(state) * SLIGHTLY_MODIFIER * feature[36],
        lambda feature, state: state**2 * 0.1 + if_(state) * SLIGHTLY_MODIFIER * feature[36],
    ),
    Task(
        "While still prioritizing all, slightly focus on those who speak Hindi.",
        lambda feature, state: state * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[12]),
        lambda feature, state: state**2 * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[12]),
    ),
    Task(
        "While still prioritizing all, slightly prioritize those who have had prior pregnancies.",
        lambda feature, state: state * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[2]),
        lambda feature, state: state**2 * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[2]),
    ),
    Task(
        "While still prioritizing all, slightly weight those who have had low education.",
        lambda feature, state: state * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[16]),
        lambda feature, state: state**2 * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[16]),
    ),
    Task(
        "While still prioritizing all, slightly focus on both the youngest and oldest by age.",
        lambda feature, state: state * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[11] or feature[7]),
        lambda feature, state: state**2 * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[11] or feature[7]),
    ),
    Task(
        "While still prioritizing all, slightly prefer the income bracket bounds for the middle 40% of the population.",
        lambda feature, state: state * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[38] or feature[39] or feature[40]),
        lambda feature, state: state**2 * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[38] or feature[39] or feature[40]),
    ),
    Task(
        "While still prioritizing all, slightly favor those women who do not own their own phone.",
        lambda feature, state: state * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[24] or feature[25]),
        lambda feature, state: state**2 * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[24] or feature[25]),
    ),
    Task(
        "While still prioritizing all, slightly prioritize impoverished younger mothers by combining the distributions of 'age' and 'education'.",
        lambda feature, state: state * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[7] and feature[16]),
        lambda feature, state: state**2 * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[7] and feature[16]),
    ),
    Task(
        "While still prioritizing all, slightly focus on the joint distribution of 'income_bracket' and 'phone_owner' for those with high financial status but no phone ownership.",
        lambda feature, state: state * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_((feature[39] or feature[40]) and (feature[24] or feature[25])),
        lambda feature, state: state**2 * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_((feature[39] or feature[40]) and (feature[24] or feature[25])),
    ),
    Task(
        "While still prioritizing all, slightly advantage those who prefer being called after 7PM 'slot' registered at an NGO.",
        lambda feature, state: state * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[31] and feature[32]),
        lambda feature, state: state**2 * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[31] and feature[32]),
    ),
    Task(
        "While still prioritizing all, slightly concentrate on mothers with several pregnancies but not much success with birth.",
        lambda feature, state: state * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[2] > 1 and feature[5] == 0),
        lambda feature, state: state**2 * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[2] > 1 and feature[5] == 0),
    ),
    Task(
        "While still prioritizing all, slightly focus on those Marathi-speakers with middle-aged mothers.",
        lambda feature, state: state * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[13] and (feature[9] or feature[10])),
        lambda feature, state: state**2 * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[13] and (feature[9] or feature[10])),
    ),
    Task(
        "While still prioritizing all, slightly emphasize beneficiaries who likely work early in the morning and late at night.",
        lambda feature, state: state * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[26] or feature[28]),
        lambda feature, state: state**2 * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[26] or feature[28]),
    ),
    Task(
        "While still prioritizing all, slightly focus on mothers at high risk due to age and number of past pregnancies.",
        lambda feature, state: state * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_((feature[10] or feature[11]) and feature[3]),
        lambda feature, state: state**2 * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_((feature[10] or feature[11]) and feature[3]),
    ),
    Task(
        "While still prioritizing all, infer technical challenges in reaching the phone that could indicate 'at-risk' beneficiaries and give slight preference.",
        lambda feature, state: state * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[24] or feature[25]),
        lambda feature, state: state**2 * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[24] or feature[25])
    ),
    Task(
        "While still prioritizing all, slightly weight those with low education",
        lambda feature, state: state * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[16]),
        lambda feature, state: state**2 * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[16]),
    ),
    Task(
        "While still prioritizing all, slightly weight the lowest income_bracket groups, the absolute lowest earners in the population.",
        lambda feature, state: state * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[35] or feature[36] or feature[37]),
        lambda feature, state: state**2 * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[35] or feature[36] or feature[37]),
    ),
    Task(
        "While still prioritizing all, slightly prefer the income_bracket bounds for the middle 40% of the population.",
        lambda feature, state: state * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[38] or feature[39] or feature[40]),
        lambda feature, state: state**2 * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[38] or feature[39] or feature[40]),
    ),
    Task(
        "While still prioritizing all, slightly advantage those who prefer being called before 10:30am 'slot' and are registered at an NGO.",
        lambda feature, state: state * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[26] and feature[32]),
        lambda feature, state: state**2 * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[26] and feature[32]),
    ),
    Task(
        "While still prioritizing all, slightly advantage those who prefer being called after the 7:30pm 'slot' and are registered as ARMMAN.",
        lambda feature, state: state * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[31] and feature[33]),
        lambda feature, state: state**2 * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[31] and feature[33]),
    ),
    Task(
        "While still prioritizing all, slightly advantage those who prefer being called between 10:30am-12:30pm and are registered at an NGO.",
        lambda feature, state: state * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[27] and feature[32]),
        lambda feature, state: state**2 * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[27] and feature[32]),
    ),
    Task(
        "While still prioritizing all, slightly advantage those who prefer being called between 12:30pm-3:30pm and are registered at an NGO.",
        lambda feature, state: state * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[28] and feature[32]),
        lambda feature, state: state**2 * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[28] and feature[32]),
    ),
    Task(
        "While still prioritizing all, slightly advantage those who are registered at an NGO and have LESS than graduate education.",
        lambda feature, state: state * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[32] and (feature[20] or feature[19] or feature[18] or feature[17] or feature[16])),
        lambda feature, state: state**2 * 0.1 + if_(state) * SLIGHTLY_MODIFIER * if_(feature[32] and (feature[20] or feature[19] or feature[18] or feature[17] or feature[16])),
    ),
]


# TODO: rerunstate**2 == state since state \in {0, 1}? potential answer: squaring the full output of the reward for shaped reward
# now we wrap the shap
def shaped_wrapper(reward_fun: RewardFun) -> RewardFun:
    def shaped_reward(feature: npt.NDArray, state: int) -> float:
        return reward_fun(feature, state) ** 2

    return shaped_reward


TASKS = []
for command in task_commands:
    command.human = shaped_wrapper(command.human)
    TASKS.append(command)

# TASKS = [TASKS[3], TASKS[6], TASKS[8], TASKS[12]]
# for t in TASKS:
#     print(t.command)


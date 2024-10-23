from dataclasses import dataclass
import dataclasses
from functools import partial, wraps
import itertools
import json
import subprocess
import os
import google.generativeai as genai
import sys

sys.path.append(".")
from rewards.tasks_list import TASKS
import time
import rewards.redis as red_db
import pickle
import os
import typer
from rich import print
from eztils.typer import dataclass_option
from eztils.persistence.hf import upload_folder
from loguru import logger
from logtail import LogtailHandler

## TODO make this configurable
logtail_handler = LogtailHandler(source_token=os.environ['LOGTAIL_SOURCE'])
logger.add(
    logtail_handler,
    format="{message}",
    level="INFO",
    backtrace=False,
    diagnose=False,
)


os.environ["EXP_ID"] = str(time.time())
# Define your parameters
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel("gemini-pro")



def update_llm_reward(new_code):
    file_path = "./rewards/rewardfun_llm.py"
    with open(file_path, "r") as file:
        lines = file.readlines()

    with open(file_path, "w") as file:
        for line in lines:
            if line.strip().startswith(
                "def llm_reward_function(state, ind, features):"
            ):
                file.write("def llm_reward_function(state, ind, features):\n")
                file.write(f"    agent_feats = features[ind, :]\n")
                file.write(f"    return {new_code}\n")
                break
            file.write(line)


def generate_new_reward_function(prompt):
    response = model.generate_content(prompt)
    print(response.text, flush=True)
    return str(response.text.split("$$$")[-2])


def choose_best_index(prompt):
    response = model.generate_content(prompt)
    print(response.text)
    best_index = int(response.text.split("index: ")[-1].strip())
    return best_index

def valid_index(index_output, max_index):
    return 0 <= index_output <= max_index
        
        
def valid_reward_function(reward_code):
    try:
        reward_fun = eval("lambda state, agent_feats: " + reward_code)
        states = [0, 1]
        agent_feats = [list(range(43))]
        for state, agent_feat in itertools.product(states, agent_feats):  # test cases
            reward_fun(state, agent_feat)
        return True
    except:
        print(f"Reward function syntax error! generated reward: {reward_code}")
        return False


def validated_gen(generate_fn, prompt, validate, max_tries=10):
    def multiple_generate_decorator(gen_fn):
        @wraps(gen_fn)        
        def wrapper(prompt):
            tries = 0
            while tries < max_tries:
                try:
                    return generate_fn(prompt)
                except:
                    tries += 1
            raise Exception("Too many generation attempts...throwing exception")
        return wrapper                
    wrapped_gen = multiple_generate_decorator(generate_fn)
    llm_output = wrapped_gen(prompt)
    tries = 0
    while not validate(llm_output):
        if tries == max_tries:
            raise Exception("Too many generation attempts...throwing exception")
        llm_output = wrapped_gen(prompt)
        tries += 1
    return llm_output



@dataclass
class Config:
    data: str = "counterexample"
    save_string: str = f"reward_{os.getenv('EXP_ID')}"
    N: int = 48
    B: float = 16.0
    robust_keyword: str = "sample_random"  # other option is "mid"
    n_train_epochs: int = 1  # 10 epoch is ok
    seed: int = 0
    cdir: str = "."
    no_hawkins: int = 1
    tp_transform: str = None
    opt_in_rate: float = 1.0
    data_type: str = "discrete"
    wandb_project: str = "llm_reward_learning"
    wandb_profile_name: str = "ezipe"
    reward_from: str = "llm" # choices=['llm', 'llm_zeroshot', 'human', 'base', 'default']
    max_tries: int = 10
    debug: bool = False
    total_splits: bool = 1
    index: int = 0
    num_stages: int = 5
    num_candidates: int = 3
    chain_of_thought: bool = True


from eztils.run_parallel import BaseHyperParameters

class HyperParameters(BaseHyperParameters):
    # TODO figure out better way to sort attrs without having to add number in beginning, and tie this so there's never a bug with reading the variables below    
    _1_arm_budget = [(48, 16)]
    # _2_arm_budget = [(48, 16)]
    _2_cot = [True, False]    
    _3_seeds = range(100, 300)
    _4_task_indices = range(len(TASKS))
    _5_n_train_epochs = [4]
    _6_llm_reward = [True, False]

hparams = HyperParameters.get_product()

def run_eval(config: Config, task_index, reward_source, stage=0, iteration=0):
    # Create the command string
    save_string = f"{config.save_string_full}_{task_index=}_{stage=}_{iteration=}"
    bash_script_eval = (
        f"bash run/armman/run.eval.run_rmabppo_armman.sh {config.cdir} {config.seed} 0 {config.data} {save_string} {config.N} {config.B} "
        f"{config.robust_keyword} {config.n_train_epochs} {config.no_hawkins} {config.tp_transform} {config.opt_in_rate} {config.data_type} {task_index} {reward_source}"
    )

    # Run the script
    subprocess.run(bash_script_eval, shell=True)


def run_train(config: Config, task_index, reward_source, stage=0, iteration=0):
    # Create the command string
    save_string = f"{config.save_string_full}_{task_index=}_{stage=}_{iteration=}"
    bash_script_train = (
        f"bash run/armman/run.train.run_rmabppo_armman.sh {config.cdir} {config.seed} 0 {config.data} {save_string} "
        f"{config.N} {config.B} {config.robust_keyword} {config.n_train_epochs} {config.no_hawkins} {config.tp_transform} {config.opt_in_rate} {config.data_type} {reward_source}"
    )

    # Run the script
    subprocess.run(bash_script_train, shell=True)

def llm_gen(
    task_index,
    command,
    config: Config,
):
    reward_from = config.reward_from
    config.save_string_full = f'{config.save_string}_rf{config.reward_from}_cot{config.chain_of_thought}_seed{config.seed}_nepochs{config.n_train_epochs}'
    
    if reward_from not in ["llm", "llm_zeroshot"]:
        run_train(config, task_index, reward_from)
        run_eval(config, task_index, reward_from)
        return

    goal_prompt = command

    if reward_from == "llm_zeroshot":
        num_stages = 1  # this many stages used for 'evolution'
        num_candidates = 1  # this many candidates generated in a stage
    else:
        num_stages = config.num_stages
        num_candidates = config.num_candidates

    best_rewards = []
    for stage in range(num_stages):
        stage_rewards = []
        red_db.set("reflections", pickle.dumps([]))

        for iteration in range(num_candidates):
            ## LLM REWARD GENERATION
            reward_history = "\n".join(
                f"- Tried: return {reward}" for reward in best_rewards
            )
            ## TODO move all prompts into separate file
            if config.chain_of_thought:
                prompt = (
                    f"Create a Python reward function for RL in phone call resource allocation to mothers in India, with the objective of prioritizing higher states and: {goal_prompt}. "
                    "The function should use 'state' (value is either 0,1) and features 'agent_feats' (length 43 array) to direct the RL agent. "
                    "Here is a description of the features you may use: "
                    "\nIndex Name DataType\n 0. Enrollment gestational age - Int\n 1. Enrollment delivery status - Int\n 2. Gravidity (number of pregnancies) - Int\n 3. Parity (number of viable pregnancies) - Int\n 4. Still births count - Int\n 5. Live births count - Int\n 6. Days to the first call - Int\n 7. Ages 10-20 - Binary\n 8. Ages 21-30 - Binary\n 9. Ages 31-40 - Binary\n 10. Ages 41-50 - Binary\n 11. Ages 51-60 - Binary\n 12. Speaks Hindi - Binary\n 13. Speaks Marathi - Binary\n 14. Speaks Gujurati - Binary\n 15. Speaks Kannada - Binary\n 16. Education level 1/7 -- illiterate - Binary\n 17. Education level 2/7 -- 1-5th Grade Completed - Binary\n 18. Education level 3/7 -- 6-9th Grade Completed - Binary\n 19. Education level 4/7 -- 10th Grade Passed - Binary\n 20. Education level 5/7 -- 12th Grade Passed - Binary\n 21. Education level 6/7 -- Graduate - Binary\n 22. Education level 7/7 -- Post graduate - Binary\n 23. Phone owner 0 (e.g., woman) - Binary\n 24. Phone owner 1 (e.g., husband) - Binary\n 25. Phone owner 2 (e.g., family) - Binary\n 26. To be called from 8:30am-10:30am - Binary\n 27. To be called from 10:30am-12:30pm - Binary\n 28. To be called from 12:30pm-3:30pm - Binary\n 29. To be called from 3:30pm-5:30pm - Binary\n 30. To be called from 5:30pm-7:30pm - Binary\n 31. To be called from 7:30pm-9:30pm - Binary\n 32. NGO - Binary\n 33. ARMMAN - Binary\n 34. PHC - Binary\n 35. Income bracket -1 (no income) - Binary\n 36. Income bracket 1 (e.g., 0-5000) - Binary\n 37. Income bracket 2 (e.g., 5001-10000) - Binary\n 38. Income bracket 3 (e.g., 10001-15000) - Binary\n 39. Income bracket 4 (e.g., 15001-20000) - Binary\n 40. Income bracket 5 (e.g., 20001-25000) - Binary\n 41. Income bracket 6 (e.g., 25001-30000) - Binary\n 42. Income bracket 7 (e.g., 30000-999999) - Binary\n "
                    "Your task:\n"
                    "1. Write a simple, single-line Python reward function. Exclude the word 'return' and exclude non-standard libraries. Format your code with triple $ signs: $$$[YOUR FUNCTION]$$$. \n"
                    #! NEXT LINE IS CHAIN OF THOUGHT
                    "2. Provide an explanation on how this function prioritizes the specified age group. Format your explanation with triple % signs: %%%[YOUR EXPLANATION]%%%. \n" 
                    "Note that HIGHER states are always preferred, so ensure reward increases as state increases. Make sure reward is always positive and increasing with state. \n"
                    "Avoid using bitwise operators &, |. Using and, or instead. \n"
                    "Example Prompt: While prioritizing all, emphasize agents that are both older and richer \n" 
                    #! NEXT LINE IS CHAIN OF THOUGHT
                    "Let's think about this step by step. We want to give reward only for agents that are older, which corresponds to feature 11, and rich which corresponds to feature 42. This corresponds to a condition of (agent_feats[11] and agent_feats[42]). In addition, we always only want to give reward when the state is 1, since the agent gets reward only when it is in a listening state. Therefore, our reward function should be: state * (agent_feats[11] and agent_feats[42]).\n"
                    "Example Response:\n"
                    "Python Code: '$$$ state * 0.1 + 2 * state * (agent_feats[11] and agent_feats[42]) $$$'\n"
                    #! NEXT LINE IS CHAIN OF THOUGHT
                    "Explanation: %%%This function gives higher rewards for higher states and higher ages, aligning with the goal to reward older individuals with higher states.%%% \n"                
                    f"Come up with a unique new reward for the specified goal: {goal_prompt}. Here are your best previous attempts: \n"
                    f"{reward_history}"
                )
            else:
                prompt = (
                    f"Create a Python reward function for RL in phone call resource allocation to mothers in India, with the objective of prioritizing higher states and: {goal_prompt}. "
                    "The function should use 'state' (value is either 0,1) and features 'agent_feats' (length 43 array) to direct the RL agent. "
                    "Here is a description of the features you may use: "
                    "\nIndex Name DataType\n 0. Enrollment gestational age - Int\n 1. Enrollment delivery status - Int\n 2. Gravidity (number of pregnancies) - Int\n 3. Parity (number of viable pregnancies) - Int\n 4. Still births count - Int\n 5. Live births count - Int\n 6. Days to the first call - Int\n 7. Ages 10-20 - Binary\n 8. Ages 21-30 - Binary\n 9. Ages 31-40 - Binary\n 10. Ages 41-50 - Binary\n 11. Ages 51-60 - Binary\n 12. Speaks Hindi - Binary\n 13. Speaks Marathi - Binary\n 14. Speaks Gujurati - Binary\n 15. Speaks Kannada - Binary\n 16. Education level 1/7 -- illiterate - Binary\n 17. Education level 2/7 -- 1-5th Grade Completed - Binary\n 18. Education level 3/7 -- 6-9th Grade Completed - Binary\n 19. Education level 4/7 -- 10th Grade Passed - Binary\n 20. Education level 5/7 -- 12th Grade Passed - Binary\n 21. Education level 6/7 -- Graduate - Binary\n 22. Education level 7/7 -- Post graduate - Binary\n 23. Phone owner 0 (e.g., woman) - Binary\n 24. Phone owner 1 (e.g., husband) - Binary\n 25. Phone owner 2 (e.g., family) - Binary\n 26. To be called from 8:30am-10:30am - Binary\n 27. To be called from 10:30am-12:30pm - Binary\n 28. To be called from 12:30pm-3:30pm - Binary\n 29. To be called from 3:30pm-5:30pm - Binary\n 30. To be called from 5:30pm-7:30pm - Binary\n 31. To be called from 7:30pm-9:30pm - Binary\n 32. NGO - Binary\n 33. ARMMAN - Binary\n 34. PHC - Binary\n 35. Income bracket -1 (no income) - Binary\n 36. Income bracket 1 (e.g., 0-5000) - Binary\n 37. Income bracket 2 (e.g., 5001-10000) - Binary\n 38. Income bracket 3 (e.g., 10001-15000) - Binary\n 39. Income bracket 4 (e.g., 15001-20000) - Binary\n 40. Income bracket 5 (e.g., 20001-25000) - Binary\n 41. Income bracket 6 (e.g., 25001-30000) - Binary\n 42. Income bracket 7 (e.g., 30000-999999) - Binary\n "
                    "Your task:\n"
                    "1. Write a simple, single-line Python reward function. Exclude the word 'return' and exclude non-standard libraries. Format your code with triple $ signs: $$$[YOUR FUNCTION]$$$. \n"
                    "Note that HIGHER states are always preferred, so ensure reward increases as state increases. Make sure reward is always positive and increasing with state. \n"
                    "Avoid using bitwise operators &, |. Using and, or instead. \n"
                    "Example Prompt: Prioritize agents that are older and rich\n"
                    "Example Response:\n"
                    "Python Code: '$$$ state * 0.1 + 2 * state * (agent_feats[11] and agent_feats[42]) $$$'\n"
                    f"Come up with a unique new reward for the specified goal: {goal_prompt}. Here are your best previous attempts: \n"
                    f"{reward_history}"
                )
                
            print(prompt)

            new_reward_function_code = validated_gen(
                generate_new_reward_function,
                prompt,
                valid_reward_function,
                config.max_tries,
            )
            print(
                f"Generated Reward Function (Validated): {new_reward_function_code}",
                flush=True,
            )

            stage_rewards.append(new_reward_function_code)
            update_llm_reward(new_reward_function_code)
            ## TRAIN + EVAL Reward
            print(f"\nIteration {iteration}: Running training", flush=True)
            run_train(config, task_index, "llm", stage, iteration)
            run_eval(config, task_index, "llm", stage, iteration)

        # EVOLUTIONARY SELECTION
        red_db.set("rewards", pickle.dumps(stage_rewards))
        # print('are you getting stuck here?')
        ref = red_db.get(("reflections")) #!! this is getting stuck...
        # print('no, not getting stuck...')
        previous_reward_reflections = pickle.loads(ref)
        
        # print('are you getting stuck here with rewards?')
        rew = red_db.get(("rewards"))
        # print('no, not getting stuck...')
        previous_rewards = pickle.loads(rew)
        
        prompt = (
            f"My goal was to create a Python reward function for RL in resource allocation, with the objective of: {goal_prompt}"
            " I tried several reward functions for this task. Below, I have the given reward function, and the corresponding distribution of reward achieved across 44 agent features."
            " A description of the features is as follows: "
            "\nIndex Name DataType\n 0. Enrollment gestational age - Int\n 1. Enrollment delivery status - Int\n 2. Gravidity (number of pregnancies) - Int\n 3. Parity (number of viable pregnancies) - Int\n 4. Still births count - Int\n 5. Live births count - Int\n 6. Days to the first call - Int\n 7. Ages 10-20 - Binary\n 8. Ages 21-30 - Binary\n 9. Ages 31-40 - Binary\n 10. Ages 41-50 - Binary\n 11. Ages 51-60 - Binary\n 12. Speaks Hindi - Binary\n 13. Speaks Marathi - Binary\n 14. Speaks Gujurati - Binary\n 15. Speaks Kannada - Binary\n 16. Education level 1/7 -- illiterate - Binary\n 17. Education level 2/7 -- 1-5th Grade Completed - Binary\n 18. Education level 3/7 -- 6-9th Grade Completed - Binary\n 19. Education level 4/7 -- 10th Grade Passed - Binary\n 20. Education level 5/7 -- 12th Grade Passed - Binary\n 21. Education level 6/7 -- Graduate - Binary\n 22. Education level 7/7 -- Post graduate - Binary\n 23. Phone owner 0 (e.g., woman) - Binary\n 24. Phone owner 1 (e.g., husband) - Binary\n 25. Phone owner 2 (e.g., family) - Binary\n 26. To be called from 8:30am-10:30am - Binary\n 27. To be called from 10:30am-12:30pm - Binary\n 28. To be called from 12:30pm-3:30pm - Binary\n 29. To be called from 3:30pm-5:30pm - Binary\n 30. To be called from 5:30pm-7:30pm - Binary\n 31. To be called from 7:30pm-9:30pm - Binary\n 32. NGO - Binary\n 33. ARMMAN - Binary\n 34. PHC - Binary\n 35. Income bracket -1 (no income) - Binary\n 36. Income bracket 1 (e.g., 0-5000) - Binary\n 37. Income bracket 2 (e.g., 5001-10000) - Binary\n 38. Income bracket 3 (e.g., 10001-15000) - Binary\n 39. Income bracket 4 (e.g., 15001-20000) - Binary\n 40. Income bracket 5 (e.g., 20001-25000) - Binary\n 41. Income bracket 6 (e.g., 25001-30000) - Binary\n 42. Income bracket 7 (e.g., 30000-999999) - Binary\n "
            "\n\nBelow are the reward functions I used and their corresponding reward distributions:\n\n"
        )
        for index, (reward, reflection) in enumerate(
            zip(previous_rewards, previous_reward_reflections)
        ):
            prompt += f"Index {index}: \nReward Function: {reward} \nReflection:\n '{reflection}'\n\n"

        prompt += (
            f"\nBased on the above reward distributions and the given goal: {goal_prompt}, please identify the index of the most effective reward function. "
            "Provide your answer EXACTLY IN the following format: 'The best reward function is at index: [INDEX]'."
        )

        print(prompt)
        
        best_index = validated_gen(choose_best_index, prompt, partial(valid_index, max_index=len(stage_rewards)), config.max_tries)

        best_rewards.append(stage_rewards[best_index])
        # breakpoint() ## NOTE: break here if you want to evaluate the reward reflections/previous rewards

    logger.info(f"Task: {task_index}. Config: {config}. Rewards: {json.dumps(best_rewards)}")



def calculate_split(total_splits, total_len, index):
    # Calculate the length of each split
    split_length = total_len // total_splits
    
    # Calculate the start and end indices of the split
    start_index = index * split_length
    end_index = start_index + split_length
    
    # Adjust the end index if the split is not evenly divided
    if index == total_splits - 1:
        end_index = total_len
    
    return start_index, end_index



def main(config: dataclass_option(Config) = "{}", wandb: bool = False):
    config: Config = config  # typehinting
    start, end =  calculate_split(config.total_splits, len(hparams), config.index)
    logger.info(f'START:{start}, END:{end}')    
    print(config)
    
    if wandb:
        import wandb as wb

        wb.init(
            project=config.wandb_project,
            entity=config.wandb_profile_name,
            name=config.save_string,
            config=dataclasses.asdict(config),
        )
        os.environ["USE_WANDB"] = "1"

    
    for ind in range(start, end):
        try:
            arm_budget, cot, seed, task_idx, n_train_epochs, llm_reward = hparams[ind]
            config.N = arm_budget[0]
            config.n_train_epochs = n_train_epochs
            config.B = float(arm_budget[1])
            config.seed = seed
            config.chain_of_thought = cot        
            task = TASKS[task_idx]        
            if llm_reward:
                config.reward_from = 'llm'
                llm_gen(task_idx, task.command, config)
            else:
                config.reward_from = 'human'
                llm_gen(task_idx, task.command, config)
                config.reward_from = 'base'
                llm_gen(task_idx, task.command, config)
                config.reward_from = 'default'
                llm_gen(task_idx, task.command, config)

            logger.info(f"Finished {ind} job!")
        except:            
            logger.error(f"Failed {ind} job!")



    logger.info(f"Finished {start}-{end} jobs!")
    try:
        upload_folder("./data", "reward_learning")
    except:
        logger.error('Upload folder did not work')


if __name__ == "__main__":
    # choose from base/human/llm_zeroshot/llm
    # goal_prompt =  "Infer disempowered mothers with little opportunity and focus heavily on them."
    # goal_prompt = "Heavily weight those who have had low education."
    # prompt = (
    #                 f"Create a Python reward function for RL in phone call resource allocation to mothers in India, with the objective of prioritizing higher states and: {goal_prompt}. "
    #                 "The function should use 'state' (value is either 0,1) and features 'agent_feats' (length 43 array) to direct the RL agent. "
    #                 "Here is a description of the features you may use: "
    #                 "\nIndex Name DataType\n 0. Enrollment gestational age - Int\n 1. Enrollment delivery status - Int\n 2. Gravidity (number of pregnancies) - Int\n 3. Parity (number of viable pregnancies) - Int\n 4. Still births count - Int\n 5. Live births count - Int\n 6. Days to the first call - Int\n 7. Ages 10-20 - Binary\n 8. Ages 21-30 - Binary\n 9. Ages 31-40 - Binary\n 10. Ages 41-50 - Binary\n 11. Ages 51-60 - Binary\n 12. Speaks Hindi - Binary\n 13. Speaks Marathi - Binary\n 14. Speaks Gujurati - Binary\n 15. Speaks Kannada - Binary\n 16. Education level 1/7 -- illiterate - Binary\n 17. Education level 2/7 -- 1-5th Grade Completed - Binary\n 18. Education level 3/7 -- 6-9th Grade Completed - Binary\n 19. Education level 4/7 -- 10th Grade Passed - Binary\n 20. Education level 5/7 -- 12th Grade Passed - Binary\n 21. Education level 6/7 -- Graduate - Binary\n 22. Education level 7/7 -- Post graduate - Binary\n 23. Phone owner 0 (e.g., woman) - Binary\n 24. Phone owner 1 (e.g., husband) - Binary\n 25. Phone owner 2 (e.g., family) - Binary\n 26. To be called from 8:30am-10:30am - Binary\n 27. To be called from 10:30am-12:30pm - Binary\n 28. To be called from 12:30pm-3:30pm - Binary\n 29. To be called from 3:30pm-5:30pm - Binary\n 30. To be called from 5:30pm-7:30pm - Binary\n 31. To be called from 7:30pm-9:30pm - Binary\n 32. NGO - Binary\n 33. ARMMAN - Binary\n 34. PHC - Binary\n 35. Income bracket -1 (no income) - Binary\n 36. Income bracket 1 (e.g., 0-5000) - Binary\n 37. Income bracket 2 (e.g., 5001-10000) - Binary\n 38. Income bracket 3 (e.g., 10001-15000) - Binary\n 39. Income bracket 4 (e.g., 15001-20000) - Binary\n 40. Income bracket 5 (e.g., 20001-25000) - Binary\n 41. Income bracket 6 (e.g., 25001-30000) - Binary\n 42. Income bracket 7 (e.g., 30000-999999) - Binary\n "
    #                 "Your task:\n"
    #                 "1. Write a simple, single-line Python reward function. Exclude the word 'return' and exclude non-standard libraries. Format your code with triple $ signs: $$$[YOUR FUNCTION]$$$. \n"
    #                 #! NEXT LINE IS CHAIN OF THOUGHT
    #                 "2. Provide an explanation on how this function prioritizes the specified age group. Format your explanation with triple % signs: %%%[YOUR EXPLANATION]%%%. \n" 
    #                 "Note that HIGHER states are always preferred, so ensure reward increases as state increases. Make sure reward is always positive and increasing with state. \n"
    #                 "Avoid using bitwise operators &, |. Using and, or instead. \n"
    #                 "Example Prompt: Prioritize agents that are older and rich \n" 
    #                 #! NEXT LINE IS CHAIN OF THOUGHT
    #                 "Let's think about this step by step. We want to give reward only for agents that are older, which corresponds to feature 11, and rich which corresponds to feature 12. This corresponds to a condition of (agent_feats[11] and agent_feats[42]). In addition, we always only want to give reward when the state is 1, since the agent gets reward only when it is in a listening state. Therefore, our reward function should be: state * (agent_feats[11] and agent_feats[42]).\n"
    #                 "Example Response:\n"
    #                 "Python Code: '$$$ state * (agent_feats[11] and agent_feats[42]) $$$'\n"
    #                 #! NEXT LINE IS CHAIN OF THOUGHT
    #                 "Explanation: %%%This function gives higher rewards for higher states and higher ages, aligning with the goal to reward older individuals with higher states.%%% \n"                
    #                 f"Come up with a unique new reward for the specified goal: {goal_prompt}. Here are your best previous attempts: \n"
    #             )    
    # for i in range(10):
    #     try:
    #         generate_new_reward_function(goal_prompt)
    #     except:
    #         pass
    #     print('*' * 100)
    #     print()
    typer.run(main)

import numpy as np
import pandas as pd
import time

def reward_reflection(sim_data, features, no_reflection=False):

    if no_reflection: 
        return "No reflection provided" 

    sim_data = sim_data[102]
    total_rewards_per_agent = np.sum(sim_data, axis=0)

    # first process integer fatures 0-6 
    def calculate_int_feature_distribution(features, total_rewards_per_agent, feature_index, ranges, feature_name):
        distribution = {}
        for low, high in ranges:
            category_name = f"{feature_name} ({low}-{high})"
            agents_in_range = np.logical_and(features[:, feature_index] >= low, features[:, feature_index] <= high)
            distribution[category_name] = np.sum(total_rewards_per_agent[agents_in_range]) / np.sum(total_rewards_per_agent) * 100
        return distribution

    # Define ranges for each integer feature
    int_feature_ranges = {
        0: [(0, 10), (11, 20), (21, 30), (31, 40)],  # Ranges for 'Enrollment gestational age'
        1: [(0, 0), (1, 1)],     # Binary ranges for 'Enrollment delivery status'
        2: [(0,0), (1, 1), (2, 4), (5, 10)],  # Ranges for 'Gravidity (number of pregnancies)'
        3: [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)],  # Distinct categories for 'Parity (number of viable pregnancies)'
        4: [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)],   # Ranges for 'Still births count'
        5: [(0, 1), (2, 4), (5, 10)],  # Ranges for 'Live births count'
        6: [(0, 30), (31, 100), (101, 200)]  # Ranges for 'Days to the first call'
    }
    int_feature_names = {
        0: "Enrollment gestational age",
        1: "Enrollment delivery status",
        2: "Gravidity (number of pregnancies)",
        3: "Parity (number of viable pregnancies)",
        4: "Still births count",
        5: "Live births count",
        6: "Days to the first call"
    }

    output_string = ""
    for feature_index, ranges in int_feature_ranges.items():
        feature_name = int_feature_names.get(feature_index, f"Unknown Feature {feature_index}")
        dist = calculate_int_feature_distribution(features, total_rewards_per_agent, feature_index, ranges, feature_name)
        output_string += f"\nCategory: {feature_name}\n"
        for category, percentage in dist.items():
            output_string += f"{category}: {percentage:.2f}%\n"

    categories = {
        "Ages": range(7, 12),
        "Income": range(35, 43),
        "Calling Times": range(26, 32),
        "Education Levels": range(16, 23),
        "Languages Spoken": range(12, 16),
        "Phone Owners": range(23, 26),
        "Organizations": range(32, 35)
    }
    feature_names = {
        7: "Ages 10-20",
        8: "Ages 21-30",
        9: "Ages 31-40",
        10: "Ages 41-50",
        11: "Ages 51-60",
        35: "Income bracket -1 (no income)",
        36: "Income bracket 1 (e.g., 0-5000)",
        37: "Income bracket 2 (e.g., 5001-10000)",
        38: "Income bracket 3 (e.g., 10001-15000)",
        39: "Income bracket 4 (e.g., 15001-20000)",
        40: "Income bracket 5 (e.g., 20001-25000)",
        41: "Income bracket 6 (e.g., 25001-30000)",
        42: "Income bracket 7 (e.g., 30000-999999)",
        26: "8:30am-10:30am",
        27: "10:30am-12:30pm",
        28: "12:30pm-3:30pm",
        29: "3:30pm-5:30pm",
        30: "5:30pm-7:30pm",
        31: "7:30pm-9:30pm",
        16: "Illiterate",
        17: "1-5th Grade Completed",
        18: "6-9th Grade Completed",
        19: "10th Grade Passed",
        20: "12th Grade Passed",
        21: "Graduate",
        22: "Post graduate",
        12: "Speaks Hindi",
        13: "Speaks Marathi",
        14: "Speaks Gujurati",
        15: "Speaks Kannada",
        23: "Phone owner - Woman",
        24: "Phone owner - Husband",
        25: "Phone owner - Family",
        32: "NGO",
        33: "ARMMAN",
        34: "PHC"
    }
    def calculate_distribution(feature_indices):
        return {feature_names.get(index, f"Unknown Feature {index}"): np.sum(total_rewards_per_agent[features[:, index] == 1]) / np.sum(total_rewards_per_agent) * 100 for index in feature_indices}
    output_string += "\n".join(f"\nCategory: {category}\n" + "\n".join(f"{feature}: {percentage:.2f}%" for feature, percentage in calculate_distribution(indices).items()) for category, indices in categories.items())
    return output_string
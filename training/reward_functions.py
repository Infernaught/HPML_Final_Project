import tasks.aime.reward_functions as aime_reward_functions
import tasks.countdown.reward_functions as countdown_reward_functions

reward_function_mapping = {
    "aime": [
        aime_reward_functions.format_reward_func,
        aime_reward_functions.returns_int_reward_func,
        aime_reward_functions.equation_reward_func,
    ],
    "countdown": [
        countdown_reward_functions.format_reward_func,
        countdown_reward_functions.equation_reward_func,
    ]
}
